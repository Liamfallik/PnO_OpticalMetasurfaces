from mp_api.client import MPRester

with MPRester(api_key='<enter your api key>') as mpr:
    bandstructure = mpr.get_bandstructure_by_material_id('mp-30')

print(bandstructure)