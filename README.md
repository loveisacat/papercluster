# PaperClustering
A demo project for research paper clustering by abstracts

:star: Please read through **'report.ipynb'** for the whole process and summary report of this project in GitHub or via Jupyter lab :blush:


:sunglasses: If you are interested in code execution, please use the following code:  
*pip install -r requirements.txt*    #For the Python environment preparation please install the required packages   
*python3 papercluster.py*            #The execution code

:boom: 'Data Prepare' tips: I developed abstractExtractor tools from multiple sources original pdf, Google Scholar or Crossref API in *'abstractextractor.py'*
For the pdf extractor you can download all pdf and put them in the folder 'paperdemo', and use the tool (https://github.com/CeON/CERMINE) to para the PDFs.  
(example: *java -cp cermine-impl-1.13-jar-with-dependencies.jar pl.edu.icm.cermine.ContentExtractor -path paperdemo -outputs "jats"*)  
I use various data source channels to ensure the data is complete and valid and output data is saved to **papers.csv**








