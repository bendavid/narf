class Dataset:
    def __init__(self, name, filepaths = [], is_data = False, xsec = None, target_lumi = None, lumi_csv = None, lumi_json = None):
        self.name = name
        self.filepaths = filepaths
        self.is_data = is_data
        self.xsec = xsec
        self.target_lumi = target_lumi
        self.lumi_csv = lumi_csv
        self.lumi_json = lumi_json
