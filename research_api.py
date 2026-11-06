"""Loopback-only API for the local SudoSleep research workspace."""
from __future__ import annotations
from functools import lru_cache
import io
import json
from pathlib import Path
import re
from threading import Lock

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from scipy.signal import spectrogram
from sklearn.metrics import roc_curve, precision_recall_curve

from dreamt_dataset import load_dreamt_segment
from wearable_pipeline import (FEATURE_NAMES, predict_with_accelerometer_checkpoint,
    apply_markov_smoothing, explain_accelerometer_model_scalars, preprocess_accelerometer,
    heuristic_scores, default_transition_matrix, make_synthetic_demo)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/dreamt/DREAMT'
DERIVED = DATA / 'derived'
CHECKPOINT = DERIVED / 'dreamt_accelerometer.pt'
app = FastAPI(title='SudoSleep local research API')
MODEL_LOCK = Lock()


def read_json(path):
    if not path.is_file():
        raise HTTPException(404, 'This research artifact is not available locally.')
    return json.loads(path.read_text())


def subject_path(subject):
    if not re.fullmatch(r'S\d{3}', subject):
        raise HTTPException(400, 'Invalid subject identifier.')
    path = DERIVED / 'features' / f'{subject}.npz'
    if not path.is_file():
        raise HTTPException(404, 'No prepared recording for this subject.')
    return path


@lru_cache(maxsize=2)
def night(subject):
    with np.load(subject_path(subject), allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def model_input(f, subject):
    return {'signals_3ch': f['signals'], 'scalars': f['scalars'],
            'labels': f['labels'], 'records': np.full(len(f['labels']), subject),
            'preprocessing_version': str(f['preprocessing_version'])}


@app.get('/api/overview')
def overview():
    report = read_json(DERIVED / 'dreamt_results.json')
    split = read_json(DERIVED / 'subject_splits.json')
    audit = pd.read_csv(DERIVED / 'audit_64hz.csv')
    preds = pd.read_csv(DERIVED / 'test_predictions.csv')
    fpr, tpr, _ = roc_curve(preds['label'], preds['score'])
    precision, recall, _ = precision_recall_curve(preds['label'], preds['score'])
    def points(x, y):
        idx = np.unique(np.linspace(0, len(x)-1, min(250,len(x))).astype(int))
        return [{'x':float(x[i]), 'y':float(y[i])} for i in idx]
    return {'report': report, 'split': split, 'status': (read_json(DERIVED / 'overnight_status.json')
                       if (DERIVED / 'overnight_status.json').is_file()
                       else {'stage': 'complete', 'updated_utc': None}),
            'audit': {'subjects':len(audit), 'samples':int(audit['rows'].sum()),
                      'hours':float(audit['elapsed_h'].sum()), 'gaps':int(audit['gaps'].sum()),
                      'invalid':int(audit['invalid_acc_rows'].sum())},
            'curves': {'roc': points(fpr,tpr), 'pr':points(recall,precision)}}


@app.get('/api/recordings')
def recordings():
    audit = pd.read_csv(DERIVED / 'audit_64hz.csv').set_index('subject')
    split = read_json(DERIVED / 'subject_splits.json')
    result = []
    for group in ['train','validation','test']:
        for s in split[group]:
            with np.load(subject_path(s), allow_pickle=False) as f:
                minutes = len(f['labels'])
            result.append({'subject':s, 'split':group, 'minutes':minutes,
                           'sleep_hours':float(audit.loc[s,'sleep_h']), 'reference_ahi':float(audit.loc[s,'AHI'])})
    return sorted(result,key=lambda x:x['subject'])


@app.get('/api/night/{subject}')
def night_overview(subject: str):
    f = night(subject)
    return {'subject':subject, 'minutes':len(f['labels']), 'timeline':[
        {'minute':i, 'target':int(label), 'sleep_fraction':round(float(f['sleep_fraction'][i]),3)}
        for i,label in enumerate(f['labels'])]}


def signal_payload(signals, start):
    names=['Motion magnitude','Respiratory proxy','Cardiac-band proxy']
    # Min/max envelope preserves transient extrema while bounding the response.
    output=[]
    for ch,name in enumerate(names):
        values=signals[:,ch,:].reshape(-1)
        step=max(1,len(values)//1800)
        rows=[]
        for i in range(0,len(values),step):
            v=values[i:i+step]
            for j in sorted(set([int(np.argmin(v)),int(np.argmax(v))])):
                rows.append({'minute':round(start+(i+j)/6000,5),'value':round(float(v[j]),5)})
        output.append({'name':name,'points':rows,'available':bool(np.any(values))})
    resp=signals[:,1,:].reshape(-1)
    freq,t,power=spectrogram(resp,fs=100,nperseg=1024,noverlap=512)
    keep=(freq>=0.05)&(freq<=1.2)
    fi=np.flatnonzero(keep); ti=np.arange(0,len(t),max(1,len(t)//180))
    spec=[[round(float(10*np.log10(power[i,j]+1e-12)),2) for j in ti] for i in fi]
    return output, {'frequency':freq[fi].tolist(),'minutes':(start+t[ti]/60).tolist(),'power':spec}


@app.get('/api/segment/{subject}')
def segment(subject: str, start: int=Query(0,ge=0), duration: int=Query(5,ge=1,le=15)):
    f=night(subject)
    stop=min(start+duration,len(f['labels']))
    if start>=stop or f['valid_seconds'][start]<30:
        raise HTTPException(400,'Choose a segment with at least 30 seconds of recorded data.')
    indices=np.arange(start,stop)
    try:
        with MODEL_LOCK:
            scores,transition=predict_with_accelerometer_checkpoint(model_input(f,subject),CHECKPOINT,indices)
        states=apply_markov_smoothing(scores,np.full(len(scores),subject),transition)
        frame=load_dreamt_segment(DATA/'data_64Hz'/f'{subject}_whole_df.csv',start*60,(stop-start)*60)
        stages=[]
        for bucket,g in frame.groupby((frame['time_s']//30).astype(int)):
            mode=g['psg_sleep_stage'].dropna().mode()
            stages.append({'minute':start+int(bucket)/2,'stage':str(mode.iloc[0]) if len(mode) else 'Unknown'})
        events=[]
        for col in frame.columns:
            if col.startswith('psg_') and col!='psg_sleep_stage':
                mask=pd.to_numeric(frame[col],errors='coerce').eq(1).to_numpy()
                changes=np.diff(np.r_[False,mask,False].astype(int))
                times=frame['time_s'].to_numpy()/60+start
                for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
                    events.append({'type':col.removeprefix('psg_').replace('_',' '),'start':float(times[a]),
                                   'end':float(times[min(b,len(times)-1)]+1/3840)})
        signals,spec=signal_payload(f['signals'][start:stop],start)
        return {'subject':subject,'start':start,'stop':stop,'backend':'CNN + Transformer',
                'signals':signals,'spectrogram':spec,'stages':stages,'events':events,
                'windows':[{'minute':int(i),'score':float(score),'state':int(state),'target':int(f['labels'][i]),
                            'features':[float(x) for x in f['scalars'][i]]} for i,score,state in zip(indices,scores,states)],
                'feature_names':FEATURE_NAMES,'note':'Smoothing is decoded within this selected segment. Labels use the provisional sparse-event interpretation.'}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f'Could not analyze this segment: {exc}') from exc


@app.get('/api/explain/{subject}/{minute}')
def explain(subject: str, minute: int):
    f=night(subject)
    if not 0<=minute<len(f['labels']):
        raise HTTPException(400,'Minute is outside this recording.')
    with MODEL_LOCK:
        names,values,base,score=explain_accelerometer_model_scalars(model_input(f,subject),CHECKPOINT,minute)
    return {'features':[{'name':n,'value':float(v)} for n,v in zip(names,values)],'base':float(base),'score':float(score)}


@app.post('/api/upload')
async def upload(file: UploadFile, sample_rate: float=Query(50,ge=8,le=1000)):
    content=await file.read(20_000_001)
    if len(content)>20_000_000:
        raise HTTPException(413,'Upload a CSV smaller than 20 MB.')
    try:
        frame=pd.read_csv(io.BytesIO(content))
        if len(frame)>sample_rate*60*15:
            raise ValueError('Choose a recording of at most 15 minutes.')
        data=preprocess_accelerometer(frame,sample_rate_hz=sample_rate)
        scores,_=heuristic_scores(data)
        signals,spec=signal_payload(data['signals_3ch'],0)
        return {'signals':signals,'spectrogram':spec,'scores':scores.tolist(),'backend':'Signal heuristic (uploaded data)'}
    except Exception as exc:
        raise HTTPException(422,str(exc)) from exc


@app.get('/api/demo')
def demo():
    data=preprocess_accelerometer(make_synthetic_demo(duration_minutes=3),sample_rate_hz=50)
    scores,_=heuristic_scores(data)
    signals,spec=signal_payload(data['signals_3ch'],0)
    return {'signals':signals,'spectrogram':spec,'scores':scores.tolist(),'backend':'Signal heuristic (synthetic data)'}


DIST=ROOT/'web/dist'
if DIST.is_dir():
    app.mount('/assets',StaticFiles(directory=DIST/'assets'),name='assets')
    @app.get('/')
    def index():
        return FileResponse(DIST/'index.html')
