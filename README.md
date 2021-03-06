# EvidenceAggregatedDriverRanking
Ranking methodology of potential Driver genes using multi-view feature sets as described in:

Mukherjee, S., Perumal, T., Daily, K., Sieberts, S., Omberg, L., Preuss, C., Carter, G., Mangravite, L. and Logsdon, B., 2019. [Identifying and ranking potential driver genes of Alzheimer's Disease using multi-view evidence aggregation](https://www.biorxiv.org/content/10.1101/534305v1). (Accepted to ISMB/ECCB 2019)

## Abstract: 

Late onset Alzheimer’s disease (LOAD) is currently a disease with no known effective treatment options. To address this, there have been a recent surge in the generation of multi-modality data (Hodes and Buckholtz, 2016; Muelleret al., 2005) to understand the biology of the disease and potential drivers that causally regulate it. However, most analytic studies using these data-sets focus on uni-modal analysis of the data. Here we propose a data-driven approach to integrate multiple data types and analytic outcomes to aggregate evidences to support the hypothesis that a gene is a genetic driver of the disease. The main algorithmic contributions of our paper are: i) A general machine learning framework to learn the key characteristics of a few known driver genes from multiple feature-sets and identifying other potential driver genes which have similar feature representations, and ii) A flexible ranking scheme with the ability to integrate external validation in the form of Genome Wide Association Study (GWAS) summary statistics. 
While we currently focus on demonstrating the effectiveness of the approach using different analytic outcomes from RNA-Seq studies, this method is easily generalizable to other data modalities and analysis types. 

We demonstrate the utility of our machine learning algorithm on two benchmark multi-view datasets by significantly outperforming the baseline approaches in predicting missing labels. We then use the algorithm to predict and rank potential drivers of Alzheimers. We show that our ranked genes show a significant enrichment for SNPs associated with Alzheimers, and are enriched in pathways that have been previously associated with the disease.


## Requirements:

This tool has been tested on Python version 2.7.13, scikit-learn version 0.19.1, numpy version 1.15.4 and scipy version 1.0.0. 


## Installation: 

```
git clone https://github.com/Sage-Bionetworks/EvidenceAggregatedDriverRanking.git
```

## Testing demo codes:
```Python
python src/Demo1.py
python src/Demo2.py
```

## Link to RNA-Seq features:
[Link to Synapse repository](https://www.synapse.org/#!Synapse:syn18097422/files/)

