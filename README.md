# DeepLearning_Crack

## Link to Dataset

[Concrete Crack Images for Classification](https://data.mendeley.com/datasets/5y9wdsg2zt/2)

There may be 2 download links as there's a *Download All 230 MB* button as well as a button to download the `Concrete Crack Images for Classification.rar` file. I've verified that both files are equal.

### What did I do? 

1. I clicked on *Download All 230 MB*
2. I downloaded it to `Dataset Folder/Original/`
3. I extracted the `5y9wdsg2zt-2.zip` and found a `Concrete Crack Images for Classification.rar` sitting inside
4. I deleted `5y9wdsg2zt-2.zip`
5. I extracted `Concrete Crack Images for Classification.rar` to `Concrete Crack Images for Classification` 
6. Updated the `.gitignore`

## What's next?

> Run the `Imbalanced Dataset Creator.ipynb`

This file is seeded to help ensure reproduciblity. The decision to keep 20% of the cracked images were based on [this research paper](https://doi.org/10.1080/09613218.2024.2321435) & [this site](https://www.holcim.com.au/products-and-services/tools-faqs-and-resources/do-it-yourself-diy/cracks-in-concrete). To be a little conservative yet still consistent with observed conditions, thought a value of 20% would be a good number to start with.

---

Next, you may run `CNN_from_scratch.ipynb` or `autoencoder.ipynb`. For the latter, ensure you configure your own `.env` file. 

--- 

Do check out the `Appendix/` folder 

## Tree

```
├───Appendix
├───Autoencoder
├───CNN
│   ├───artifacts
│   │   └───plots
├───Dataset Folder
│   ├───Actual
│   │   └───Concrete Crack Images for Classification
│   │       ├───Negative
│   │       ├───Positive
│   │       └───Unseen
│   └───Original
│       └───Concrete Crack Images for Classification
│           ├───Negative
│           └───Positive
└───ViT
```