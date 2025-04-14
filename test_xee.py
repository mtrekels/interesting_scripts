import ee
import xarray

ee.Initialize(
    project='nithecs-436810',
    opt_url='https://earthengine-highvolume.googleapis.com')


ic = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterDate(
    '2021-03-05', '2021-03-31')
ds = xarray.open_dataset(ic, engine='ee', crs='EPSG:4326', scale=0.25)

print(ds)