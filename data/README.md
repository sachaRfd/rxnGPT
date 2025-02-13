# Downloading the datasets:


- Reaction Data can be found at https://ibm.ent.box.com/v/MolecularTransformerData/data
- Best to download the zip file within this directory


To Unzip the file simply run:

```bash
unzip -o data.zip -d .
```

Move all the files within the nested data into the actual data dir :).


Next, to create the vocabulary for the datasets, please run:

```bash
    python get_vocab.py /path/to/dataset
```

