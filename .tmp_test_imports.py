import importlib, sys

try:
    importlib.import_module('src.llm_enhancement')
    print('Imported src.llm_enhancement OK')
except Exception as e:
    print('Error importing src.llm_enhancement:', e)

try:
    importlib.import_module('llm_enhancement')
    print('Imported root llm_enhancement OK')
except Exception as e:
    print('Error importing root llm_enhancement:', e)
