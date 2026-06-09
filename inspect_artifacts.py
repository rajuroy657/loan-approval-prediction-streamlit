import joblib
import os

print('cwd', os.getcwd())
artifacts = joblib.load('model_prod_files.pkl')
print('type', type(artifacts))
print('keys', list(artifacts.keys()) if hasattr(artifacts, 'keys') else 'no_keys')
print('model', type(artifacts.get('model', None)))
