Annotation Guidelines
=====================

* [index](./index.md)

Entity Type Definitions for NLP Recognition

### "tissue"
#### Definition: Supra-cellular anatomical entities (types of parts of the body and body substances, anatomical boundaries and spatial regions) from organisms within the Metazoa taxon (NCBITaxon:33208). 
* Please, not that the definition above is a *very broad interpretation* of this entity type that includes all types of tissues, organs, organ systems or any other anatomical structures, spaces or body parts.
* The "supra-cellular" restriction distinguishes these tissue entities from their constituent "cell types", because they represent a higher level of organisation of cells together and connective tissues. 
* The intended meaning of "tissue" in the present context does not include instances of excised pieces of anatomical parts of a body, *i.e.* a biopsy, specimen, or tissue sample, but rather it refers to the type of their anatomical site of origin. 
* Note that there is some overlap between the use of the word "tissue" meaning a piece of excised tissue (a "biopsy", a "sample", i.e. some part of an anatomical entity), and a description which represents the source tissue (for example, a piece of tissue that is being used to characterise the cell type composition of the source tissue). 
* When labelling entities, include adjectives, descriptors in the text span referring to an entity that can be seen as part of a descriptive phrase that identifies a set of anatomical entities / distinguishes one group of anatomical entities from another similar group. 
* Example annotations include:
    * "immune system", "spleen", "blood", "plasma", "blood vessels", "retinal pigment epithelium"
    * Examples of text spans NOT annotated as "tissue":
    * "Duodenal biopsy samples …", but you would annotate "duodenum" in a context of "Intestinal biopsy samples were collected at endoscopy from duodenum."
    * "Intestinal biopsy samples …"
* These annotations are to exclude terms which are part of a larger entity of a separate entity type. Prime examples of this would include tissues used in descriptors of indications like tumors / cancer, e.g. "lung adenocarcinoma". Another example would be the title of a database which contains a tissue name, e.g. "The Human Lung Cell Atlas". Biological process names that contain anatomical names (e.g. "heart contraction"); disease names that contain anatomical terms (e.g. "lung cancer") should be also excluded.

#### Anatomical entity references
* Anatomy textbooks, anatomy atlases
* Uberon multi-species anatomy ontology [(Uberon)](https://www.ebi.ac.uk/ols4/ontologies/uberon)
    * Mungall CJ, Torniai C, Gkoutos GV, Lewis SE, Haendel MA. Uberon, an integrative multi-species anatomy ontology. Genome Biology. 2012 Jan;13(1):R5. DOI: 10.1186/gb-2012-13-1-r5. PMID: 22293552; PMCID: PMC3334586.
    * Haendel MA, Balhoff JP, Bastian FB, et al. Unification of multi-species vertebrate anatomy ontologies for comparative biology in Uberon. Journal of Biomedical Semantics. 2014 ;5:21. DOI: 10.1186/2041-1480-5-21. PMID: 25009735; PMCID: PMC4089931.
* Experimental Factor Ontology (EFO) - [EFO:0000786 anatomy basic component](https://www.ebi.ac.uk/ols4/ontologies/efo/classes/http%253A%252F%252Fwww.ebi.ac.uk%252Fefo%252FEFO_0000786)
* [Medical Subject Headings](https://meshb.nlm.nih.gov) (MeSH Concepts and synonyms, MeSH TopicalDescriptors)
* [Terminologia Anatomica](https://ta2viewer.openanatomy.org/) (TA; a standard by the International Federation of Associations of Anatomists)
* NCI Thesaurus - [NCIT:C12219 Anatomic Structure, System, or Substance](https://www.ebi.ac.uk/ols4/ontologies/ncit/classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FNCIT_C12219?lang=en)
* Foundational Model of Anatomy Ontology [(FMA)](https://www.ebi.ac.uk/ols4/ontologies/fma)
* wikipedia - [Anatomy](https://en.wikipedia.org/wiki/Anatomy)
* SNOMED CT ("Body Structure")

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

### "cell type"
#### Definition: In vivo cell types that constitute the anatomical entities of Metazoan organisms (NCBITaxon:33208) and can be classified as a subclass of CL:0000000 cell.
Entities in this category encompass any classification categories of the physical entities that biologists call cells. You can categorise cells based on size, shape, histological staining, tissue origin, developmental stage, etc. Cell states in the sense of cells being at a particular stage of a biological process including the expression of a specific set of genes (transcriptional state) can be also used for categorising cells, thus they are regarded as subcategories of cell types for entity recognition purposes. When labelling entities, include adjectives, descriptors in the text span referring to an entity that can be seen as part of a descriptive phrase that identifies a set of cells and/or distinguishes a set of cells from other similar cells.
* Examples of identified text spans of this type include:
    * "CD4+ T cells",
    * "Th17 cells",
    * "T cells",
    * "neutrophils",
    * "neurons".

#### Cell type references
* Cell Ontology [(CL)](https://www.ebi.ac.uk/ols4/ontologies/cl)
    * Tan SZK, Puig-Barbe A, Goutte-Gattat D, et al. The Cell Ontology in the age of single-cell omics. ArXiv [Preprint]. 2025 Jun 17:arXiv:2506.10037v2. PMID: 40735089; PMCID: PMC12306828.
    * Meehan TF, Masci AM, Abdulla A, et al. Logical development of the cell ontology. BMC Bioinformatics. 2011 Jan;12:6. DOI: 10.1186/1471-2105-12-6. PMID: 21208450; PMCID: PMC3024222.
* Experimental Factor Ontology (EFO) - [CL:0000000 cell](https://www.ebi.ac.uk/ols4/ontologies/efo/classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FCL_0000000)
* Provisional Cell Ontology [(PCL)](https://www.ebi.ac.uk/ols4/ontologies/pcl)
* [Medical Subject Headings](https://meshb.nlm.nih.gov) (MeSH Concepts and synonyms, MeSH TopicalDescriptors)
* NCI Thesaurus - [NCIT:C12508 Cell](https://www.ebi.ac.uk/ols4/ontologies/ncit/classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FNCIT_C12508)
* wikipedia - [Cel type](https://en.wikipedia.org/wiki/Cell_type)
SNOMED CT ("cell")

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

### "cell line"
#### Definition: Immortalised cultured cell lines (CLO:0000019) derived from Metazoan organisms (NCBITaxon:33208). These are cultured cells (CL:0000010) that are also immortal cell line cells. When labelling entities, include adjectives, descriptors in the text span referring to an entity that can be seen as part of a descriptive phrase that identifies a set of cell lines and/or distinguishes a set of cell lines from other similar cell lines.
Examples of identified text spans of this type include: "MCF-7", "HeLa", "HepG2"

#### Cell line references
* American Type Culture Collection [(ATCC)](https://www.atcc.org/)
* [Cellosaurus](https://www.cellosaurus.org/) - Cell line encyclopedia
    * Bairoch A. The Cellosaurus, a Cell-Line Knowledge Resource. J Biomol Tech. 2018 Jul;29(2):25-38. [doi: 10.7171/jbt.18-2902-002](https://doi.org/10.7171/jbt.18-2902-002). Epub 2018 May 10. PMID: 29805321; PMCID: PMC5945021.
* Cell Line Ontology [(CLO)](https://www.ebi.ac.uk/ols4/ontologies/clo/classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FCLO_0000031?lang=en)
    * Sarntivijai S, Lin Y, Xiang Z, et al. CLO: The cell line ontology. Journal of Biomedical Semantics. 2014 ;5:37. [DOI: 10.1186/2041-1480-5-37](https://doi.org/10.1186/2041-1480-5-37). PMID: 25852852; PMCID: PMC4387853.

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

### Vague or ambiguous entities

Descriptors of terms extra to those defined above, deemed too vague to classify as a true entity type "tissue" or "cell type". These descriptions aid us in understanding our annotation logic and allow for the production of more ML-usable data.
* "vague tissue": A sub-type of "tissue" (as defined above) that encompasses too wide a range of tissues, anatomical entities / structures to be useful for the annotation of experimental data used in the ML model production or in knowledge bases. It may or may not be clear what specific anatomical entity a "vague tissue"-type mention refers to in the broader context of the paper text it is found in.
    * Examples of identified text spans of this type include: "non-lymphoid tissues", "cancer tissue".
* "vague cell type": A cell type (as defined above) that maps to an overly broad category, making it too non-specific to be used for the annotation of experimental data used in the ML model production or in knowledge bases. It may or may not be clear what specific anatomical entity a "vague cell type" refers to in the broader context of the paper text it is found.
    * Examples of identified text spans of this type include: "cells", "infected cells", "control cells".
    * Additional examples: [vague_entity_examples.tsv](./vague_entity_examples.tsv])

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

