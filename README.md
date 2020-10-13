# odat

Tools to access raw and prepared sound/vibration data.

Modules for prepping data, on a per data/project basis, 
keeping modules self-contained as much as possible.

# Examples

A `dacc` is a data-accessor. It has methods that provide you with data, ready to use.

## List the daccs you have access to
 
```pydocstring
>>> from odat import daccs  # if code changed (e.g. modules added), need to restart kernel to get new daccs
>>> list(daccs)
['xe', 'iatis', 'conveyor_belts_01']
```

Note: If a dataset doesn't show up, it's the there was an error loading the dataset module.
Most of the time the error is that you don't have the credentials. 
But it could be our fault too!
See the Troubleshooting section below for more information.

## Print the list, with descriptions

```pydocstring
>>> print(*(f"{name}\n\t{info.get('description', 'no description')}" for name, info in daccs.items()), sep='\n')
xe
	Fridge compressor data
iatis
	Over 5 million tagged sounds
conveyor_belts_01
	Conveyor belts
```

## Get a dacc

```pydocstring
>>> daccs['xe']
{'name': 'xe',
 'description': 'Fridge compressor data',
 'mk_dacc': <function odat.mdat.xe.mk_dacc(...)>}
```

So I get a `mk_dacc`, that apparently is a function to actually get the 
`dacc` for me:

```pydocstring
>>> mk_dacc = daccs['xe']['mk_dacc']
>>> dacc = mk_dacc()  # most of the time all args have defaults for ease of use, but can be configured
```

## What can I do with a dacc

Well let's see what attributes it has:

```pydocstring
>>> dacc = daccs['iatis']['mk_dacc']()
>>> from odat import print_attrs
>>> print_attrs(dacc)
- cache_sref_tag_df:
	bool(x) -> bool Returns True when the argument x is true, False otherwis...
- data_rootdir:
	str(object='') -> str str(bytes_or_buffer[, encoding[, errors]]) -> str...
- djoin
- fv_mgc_store:
	Local files store for text dataData is assumed to be a JSON string, and i...
- random_wfsrs_tagged
- sref_tag_bytes_gen
- sref_tag_df:
	Two-dimensional, size-mutable, potentially heterogeneous tabular data....
- sref_tag_store
- sref_tag_wfsr_gen
- tag_counts:
	One-dimensional ndarray with axis labels (including time series). La...
- tag_wf_gen
- tagged_fvs_of_users
- wf_tag_gen:
	Get a (wf, tag) pairs iterator for given tags
```

Note that if no docs is given for an attribute, you'll get something more like this:

```pydocstring
>>> dacc = daccs['se']['mk_dacc']()
>>> from odat import print_attrs
>>> print_attrs(dacc)
- chk_tag_gen
- chk_tag_pairs
- fv_tag_pairs
- key_chks_gen
- key_filt
- key_fvs_gen
- key_snips_gen
- key_tag_chks_gen
- key_tag_fvs_gen
- key_tag_snips_gen
- key_tag_wf_gen
- key_to_tag
- key_wf_gen
- kv_store
- wf_tag_gen
```

That `wf_tag_gen` is quite common. 
It gives us an iterator of `(wf, tag)` pairs where `wf` is a waveform (as a numerical array),
and `tag` is an annotation, or label, describing some aspect of that waveform.

You usually use it in this form:

```pydocstring
for wf, tag in dacc.wf_tag_gen():  # here the wf_tag_gen takes no arguments (but sometimes args are required)
    # do something cool
```
# Troubleshooting

## When you don't see a dacc you expect

You want to see why some of the modules didn't show up in your daccs dict:

```pydocstring
>>> from odat import dacc_info_gen
>>> _ = list(dacc_info_gen(on_error='print'))  # or on_error='raise' if you want to blow up on the first error and see stacktrace
Error with odat.mdat.sa: 'context.csv'
```

You want more details on a specific module (here, the `sa` module) that has errors:

```pydocstring
>>> from odat.mdat import sa
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
...
KeyError: 'context.csv'
```