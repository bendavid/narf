# narf
narf is not an rdf framework

## Instructions:

Easiest way to get an appropriate build of ROOT and other dependencies is to use the singularity image:
```bash
singularity run /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling\:latest
```

Setup the environment (just adds narf to PYTHONPATH)
```bash
source setup.sh
```

Run the example
```bash
python test.py
```
