# finemap-godmc
Repo for fine-mapping the GoDMC mQTL dataset using the pan-UKB LD reference panel.

## Pre-processing steps

Following the Finucane lab GTEx eQTL fine-mapping:
 - Filter out CpGs with no genome-wide significant associations ($p< 5 \times 10^{-8}$)
 - Filtering out the MHC due to high LD

Also removed indels to only retain SNPs.

## Fine-mapping process

For each CpG:
 - Filter QTL dataset for SNPs acting upon the CpG.
 - Collect all variants within a 3Mb window of the lead SNP.
 - Collect the LD matrix for these variants and symmetrise.
 - Run `susie_rss` from `susieR` using the Z-scores and LD matrix, setting L to 8.
 - Collect PIPs and mean posterior effect sizes.

## LD reference panel

Currently using the UKBB European LD reference panel, accessed using hail.

This may not be the best reference panel as GoDMC has many data sources, with the majority of data overall being European descent split between UK and Netherlands.

We don't have a direct reference panel for the dataset, but this should be a reasonable proxy to get preliminary results (?).
