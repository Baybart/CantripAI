import json
import os





class DataLoader:

    def __init__(self, project_root):

        # script_dir = os.path.dirname(os.path.abspath(__file__))
        # project_root = os.path.abspath(os.path.join(script_dir, '..'))

        self.project_root = project_root
        self.aligned_data_path = os.path.join(project_root, 'data', 'CRD3', 'aligned data')

        self.train_files = []

    def train_files_fetch(self, train_ct):
        train_eps = []

        with open(os.path.join(self.aligned_data_path, 'train_files'), 'r', encoding='utf-8') as f:
            train_eps_string = f.read()
            train_eps_string = train_eps_string.split()
            for train_ep in train_eps_string:
                train_eps.append(train_ep)
                train_ct-=1
                if train_ct == 0:
                    break
            
            for train_ep in train_eps:
                train_ep_files = [f for f in os.listdir(os.path.join(self.aligned_data_path, 'c=4')) if f.startswith(train_ep) and os.path.isfile(os.path.join(os.path.join(self.aligned_data_path, 'c=4'), f))]
                self.train_files.extend(train_ep_files)

    def parse_json(self, f):
        data = json.load(f)

        for item in data:
            # Optionally yield summary chunks
            chunk = item.get("CHUNK", "").strip()
            if chunk:
                yield chunk

            # Extract dialogue text from turns
            turns = item.get("TURNS", [])
            for turn in turns:
                utterances = turn.get("UTTERANCES", [])
                if utterances:
                    paragraph = " ".join(u.strip() for u in utterances if u.strip())
                    if paragraph:
                        yield paragraph

    def train_iterator(self):
        for filename in self.train_files:
            filepath = os.path.join(self.aligned_data_path, 'c=4', filename)
            if os.path.isfile(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    yield from self.parse_json(f)