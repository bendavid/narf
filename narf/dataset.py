class Dataset:
    def __init__(self, name, filepaths = [], is_data = False, xsec = None, lumi_csv = None, lumi_json = None):
        self.name = name
        self.filepaths = filepaths
        self.is_data = is_data
        self.xsec = xsec
        self.lumi_csv = lumi_csv
        self.lumi_json = lumi_json
