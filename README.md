# Interactive Visualization of BERT Embeddings

This creates interactive plots of BERT word embeddings using PyTorch.  Utilizing UMAP dimension reduction, the high dimension word vectors created with BERT can be visualized in 2 dimensions.  The code works, but a bit messy (sorry for that).

### Prerequisites

The following packages are needed: pytorch, numpy, pandas, seaborn, matplotlib, nltk, umap, and bokeh.  These can be installed using pip.  For example:

```
pip install pandas
```

### Included Files

TestEmbeddings.py contains a function to scan a dataset for sentences containing a specific word, then return the BERT word vectors for the specified word in each sentence.

UmapBert.py contains the fuction to create the interactive plot given the word embeddings and the sentences.

in_domain_train.tsv is the datset used for this example.  It contains the CoLA dataset.

## Sample Output

Here is a sample interactive plot created with the word "up".  The program scans a dataset (in this example, CoLA) for many sentences containing the word "up" and gets the BERT word embeddings.  Then it plots the UMAP representation in 2D.

When you hover over each point in the interactive plot, you will get information (Such as the sentence each word vector came from).  As you can see, when using BERT, the same word can have different vectors depending on the context.

Interestingly, the emeddings of words used in a similar context lie near eachother.  Additionally, when used in different context, the word vectors are far apart.  For example, the plot below contains 4 distinct groups, each group representing distinct contexts of the word "up".


Image of a sample plot created with this program.  Notive how a pop up with information about the point appears when the cursor hovers over a point.

!["Image of an interative plot created with this program.  Notive how a pop up with information about the point appears when the cursor hovers over the point."](/sample_interactive_plot_image.png)
