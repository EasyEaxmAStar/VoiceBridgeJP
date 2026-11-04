"""Optional pinned native generation controls; production defaults unchanged."""
DEFAULT_GENERATION=dict(text_split_method='cut5',batch_size=1,streaming_mode=True,
    parallel_infer=True,min_chunk_length=16,overlap_length=2)


def generation_parameters(value=None):
    result={**DEFAULT_GENERATION,**dict(value or {})}
    if set(result)!=set(DEFAULT_GENERATION):raise ValueError('Unknown native generation option')
    if result['text_split_method'] not in {'cut0','cut3','cut5'}:raise ValueError('Unsupported native text split method')
    for key in ['streaming_mode','parallel_infer']:
        if type(result[key]) is not bool:raise ValueError('Native '+key+' must be boolean')
    for key,allowed in [('batch_size',{1,2,4}),('min_chunk_length',{16,32,64,128}),('overlap_length',{2,4,8})]:
        if type(result[key]) is not int or result[key] not in allowed:raise ValueError('Unsupported native '+key)
    return result
