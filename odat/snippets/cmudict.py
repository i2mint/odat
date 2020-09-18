# Note: You may want to use cmudict instead (pip install cmudict -- https://github.com/prosegrinder/python-cmudict)
#   (Unless you want to parse the raw data yourself)

url = 'https://github.com/Alexir/CMUdict/raw/master/cmudict-0.7b'


def fetch_zip_and_save(save_to_dir,
                       zip_filename='cmudict_raw_data.zip',
                       inner_zipfilename='decode_me_with_latin1.txt'):
    import requests

    r = requests.get(url)
    if r.status_code == 200:
        from py2store.slib.s_zipfile import ZipStore
        save_zipfilepath = os.path.join(save_to_dir, zip_filename)
        ZipStore(save_zipfilepath)[inner_zipfilename] = r.content
    else:
        raise ValueError(
            f"response status code was {r.status_code} and content {r.content}")
