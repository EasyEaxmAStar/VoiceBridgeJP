"""Optional official v2ProPlus sampling; production defaults remain unchanged."""
import math

DEFAULT_SAMPLING=dict(top_k=15,top_p=1.0,temperature=1.0,repetition_penalty=1.35,seed=42)


def sampling_parameters(value=None):
    result={**DEFAULT_SAMPLING,**dict(value or {})}
    if set(result)!=set(DEFAULT_SAMPLING):raise ValueError('Unknown native TTS sampling parameter')
    if type(result['top_k']) is not int or not 1<=result['top_k']<=100:raise ValueError('top_k must be integer1..100')
    if type(result['seed']) is not int or not -1<=result['seed']<2**31:raise ValueError('seed must be -1(stochastic) or integer0..2^31-1')
    for key,low,high in [('top_p',.05,1.),('temperature',.1,2.),('repetition_penalty',1.,2.)]:
        if type(result[key]) not in (int,float) or not math.isfinite(result[key]) or not low<=result[key]<=high:
            raise ValueError('Invalid native TTS '+key)
    return result
