# POLITICAL ANALYSIS SYSTEM
## INFORMATION RETRIEVAL: FINAL PROJECT

Automated text analysis techniques have taken on an increasingly important role in the study of political parties and political discourse. Researchers have studied manifestos, parliamentary speeches, and debates at national party meetings. These methods have proven very promising for measuring latent characteristics of texts. However, in their application, scaling models require many decisions by the researcher that likely have substantial implications for the analysis. María Ramos is dedicated to the analysis of political texts and has hired you to perform an analysis of political texts. She thinks a starting point could be the presidential reports from Carlos de Salinas to the present. These reports can be found online. However, the system should not be restricted to these and must be usable with any political text.

She is interested in creating graphs and knowing, for example, which topics each of the presidents (authors) under study addressed the most. Whether they talked about improving wages, oil profits, health, etc.

She is also interested in identifying similarities and differences between representatives of different parties. María has considered the following functions for her system:

### General Statistics
Here, general data of the documents such as the length of each document (number of words), the number of different words in the texts, the number of documents analyzed by politician, and other useful statistics will be presented.

### Lexical Dispersion Plot
The importance of a word/token can be estimated by its dispersion in the corpus. Tokens vary in their distribution throughout the text, indicating when or where certain tokens are used in the text. Lexical dispersion is a measure of the homogeneity of the word throughout the corpus. Word distributions can be generated to get a general idea of the topics, their distribution, and their changes. A lexical dispersion plot represents the occurrences of the word and the frequency with which they appear from the beginning of the corpus. Therefore, lexical dispersion diagrams are useful for identifying patterns.

The x-axis of the lexical dispersion plot shows the word offset, i.e., the appearance and frequency of words throughout the speeches, and the y-axis shows the specific issues.

### Time Series Models
In predictive analysis, time is a very important factor to consider. The recognized or predicted pattern must be studied and verified concerning time. Time series data is simply a series of data ordered over time. Figure 5 captures the comparative trend of the time series between the topics “peace” and “terrorism” as an example.

### WordCloud Representation of Speeches
To get a quick and holistic impression of the speech transcripts under consideration, word clouds are created. Word clouds are a simple and intuitive visualization technique often used to provide a first impression of text documents. WordClouds display the most frequent words in the text as a weighted list of words in a specific spatial layout, for example, sequential, circular, or random layout. The font sizes of the words change according to their relevance and frequency of appearance, and other visual properties such as color, position, and orientation often vary for aesthetic reasons or to visually encode additional information.

### Examine Trends/Patterns Using Bar Charts
Bar charts represent how data is distributed among certain potential values. Although a simple-looking chart, a bar chart has the ability to capture the essence of the data by judging dispersion and answering certain questions. The figure below is a representation of "Topic name versus number of mentions." For each provided token name, the frequency of occurrence is calculated and the chart is generated.

### Classifier
María wants to characterize writings of Mexican politics as Neoliberal and belonging to Mexican Humanism. Therefore, she also wants the system to be able to differentiate them. María requests a proposal on what her system could provide according to the functions she describes. She also hopes, if possible, that you can provide additional functions that may be useful to her, for example, concept detection. She is open to any proposal.
