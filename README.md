# LDA in arXiv
applying LDA to the arXiv full article database 

## step 1: download
following the arXiv [Documentation on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv) to have bullk acces to the PDF's from the cloud storage.

```
# Download all the Computer Science (cs) PDF source files
gsutil cp -r gs://arxiv-dataset/arxiv/cs/pdf  ./a_local_directory/
```
## step 2: set-up GROBID
according to [it's own doc's](https://grobid.readthedocs.io/en/latest/Introduction/), "GROBID is a machine learning library for extracting, parsing and re-structuring raw documents such as PDF into structured XML/TEI encoded documents". 

To work with GROBID we need to:
  1. install and Set-up the [GROBID java REST service](https://github.com/kermitt2/grobid#latest-version) (preferably in a high-memory server).
  2. install the [Grobid python client](https://github.com/kermitt2/grobid_client_python), to acces the service via python.

## step 3: run GROBID
to run grobid, you first need to star the GROBID REST service, by running the following commands on the grobid directory:

```
#open a new screen named 'GROBID'
screen -S GROBID

#start GROBID service then exit the screen
./gradlew run
```

after running the server you may use the Grobid python client to parse the pdf's with the 'processFulltextDocument' command and omitting the '--output' to create the tei.xml files alongside the pdf's:

```
grobid_client --input ~/your_pdf_directory processFulltextDocument
```

## step 3: search, clean and run LDA (under construction)
at this step we use a packege of functions that handle the data in the .tei.xml files created with grobid, that search phrases containing specific terms and then running LDA on thos phrases. 

```
from functions import * 
test_final('your__testfile_name')
```

