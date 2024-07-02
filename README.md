# PaperClustering
A demo project for research paper clustering
(This is a demo project to cluster the research papers by abstracts)

Please read through the process and summary report of the results of this project 
directly in 'report.ipynb' via Jupyter lab.


If you are interested in code execution, please read the following:  
For the Python environment preparation please install the required packages.  
pip install -r requirements.txt  
The execution code: python3 papercluster.py

'Data Prepare' tips: I developed abstractExtractor tools from multiple sources original pdf, Google Scholar or Crossref API in 'abstractextractor.py'
For the pdf extractor you can download all pdf and put them in the folder 'paperdemo', and use the tool (https://github.com/CeON/CERMINE) to para the PDFs.  
(example: java -cp cermine-impl-1.13-jar-with-dependencies.jar pl.edu.icm.cermine.ContentExtractor -path paperdemo -outputs "jats")
I use various data source channels to ensure that the data is complete and valid. The output data is saved to papers.csv








