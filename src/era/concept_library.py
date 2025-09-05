import yaml

class ConceptLibrary:
    def __init__(self, path: str):
        with open(path, 'r') as f:
            self.data = yaml.safe_load(f)
    def domains(self):
        return [d['name'] for d in self.data.get('domains', [])]
    def iter_concepts(self):
        for d in self.data.get('domains', []):
            for c in d.get('concepts', []):
                yield d['name'], c['name'], c
