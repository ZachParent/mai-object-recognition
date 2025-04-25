[[SegmentAnything_2023.pdf]]
# intro

> ([[SegmentAnything_2023.pdf#page=1&selection=133,10,146,64|SegmentAnything_2023, p.1]])
> We aim to build a foundation model for segmentation by introducing three interconnected components: **a promptable segmentation task**, a **segmentation model (SAM)** that powers data annotation and enables zero-shot transfer to a range of tasks via prompt engineering, and a data engine for collecting SA-1B, our **dataset of over 1 billion masks**

> ([[SegmentAnything_2023.pdf#page=1&selection=162,58,166,34|SegmentAnything_2023, p.1]])
> e are releasing the Segment Anything Model (SAM) and corresponding dataset (SA-1B) of 1B masks and 11M images at https://segment-anything.com to foster research into foundation models for computer vision.
## task
> ([[SegmentAnything_2023.pdf#page=1&selection=170,0,182,27|SegmentAnything_2023, p.1]])
> **Large language models** pre-trained on web-scale datasets are revolutionizing NLP with *strong zero-shot and few-shot generalization* [10]. These “foundation models” [8] can generalize to tasks and data distributions beyond those seen during training. This capability is often implemented with prompt engineering in which hand-crafted text is used to prompt the language model to generate a valid textual response for the task at hand

> ([[SegmentAnything_2023.pdf#page=1&selection=193,0,203,17|SegmentAnything_2023, p.1]])
> For example, CLIP [82] and ALIGN [55] use *contrastive learning to train text and image encoders that align the two modalities*. Once trained, engineered text prompts enable zero-shot generalization to novel visual concepts and data distributions. **Such encoders also compose effectively with other modules to enable downstream tasks, such as image generation (e.g., DALL·E [83]).** 

> ([[SegmentAnything_2023.pdf#page=2&selection=57,11,62,6|SegmentAnything_2023, p.2]])
> The requirement of a valid output mask means that **even when a prompt is ambiguous and could refer to multiple objects** (for example, a point on a shirt may indicate either the shirt or the person wearing it), *the output should be a reasonable mask for at least one of those objects*.


### ⭐ questions
> ([[SegmentAnything_2023.pdf#page=1&selection=227,0,241,30|SegmentAnything_2023, p.1]])
> 1. What task will enable zero-shot generalization?
> 2. What is the corresponding model architecture?
> 3. What data can power this task and model?

## model
### requirements
> ([[SegmentAnything_2023.pdf#page=2&selection=70,6,82,1|SegmentAnything_2023, p.2]])
> In particular, the model must support *flexible prompts*, needs to compute masks in amortized *real-time* to allow interactive use, and must be *ambiguity-aware*.
### design
> ([[SegmentAnything_2023.pdf#page=2&selection=84,0,87,33|SegmentAnything_2023, p.2]])
> a powerful **image encoder** computes an image embedding, a **prompt encoder** embeds prompts, and then the *two information sources are combined* in a **lightweight mask decoder** that predicts segmentation masks.

1. flexible prompts
	- separating the image encoder from prompt encoder, allows reusing the image for multiple prompts
	- focus on point, box, mask prompts
		- initial results for text prompts
1. real-time
	- the prompt encoder and mask decoder predict a mask in ~50ms
2. ambiguity-aware
	- SAM predicts multiple masks for a single prompt, to handle ambiguity
### evaluation
> ([[SegmentAnything_2023.pdf#page=2&selection=185,36,189,28|SegmentAnything_2023, p.2]])
> using a diverse new suite of 23 segmentation datasets, we find that SAM *produces high-quality masks from a single foreground point*, often only slightly below that of the manually annotated ground truth.

> ([[SegmentAnything_2023.pdf#page=2&selection=189,29,194,39|SegmentAnything_2023, p.2]])
> Second, we find consistently strong quantitative and qualitative results on a variety of *downstream tasks under a zero-shot transfer protocol using prompt engineering*, including **edge detection**, **object proposal generation**, **instance segmentation**, and a preliminary exploration of **text-to-mask prediction**.
## dataset
> ([[SegmentAnything_2023.pdf#page=2&selection=21,0,30,46|SegmentAnything_2023, p.2]])
> Unfortunately, there is no web-scale data source for segmentation; to address this, we build a **“data engine”**, i.e., *we iterate between using our efficient model to assist in data collection and using the newly collected data to improve the model*. We introduce each interconnected component next, followed by the dataset we created and the experiments that demonstrate the effectiveness of our approach.
### data engine
1. assisted-manual
2. semi-automatic
3. fully automatic
> ([[SegmentAnything_2023.pdf#page=2&selection=135,1,149,33|SegmentAnything_2023, p.2]])
> In the first stage, *SAM assists annotators in annotating masks*, similar to a classic interactive segmentation setup. In the second stage, *SAM can automatically generate masks for a subset of objects* by prompting it with likely object locations and **annotators focus on annotating the remaining objects**, helping increase mask diversity. In the final stage, we *prompt SAM with a regular grid of foreground points*, yielding on average **∼100 high-quality masks per image**.
### SA-1B
> ([[SegmentAnything_2023.pdf#page=2&selection=162,19,169,14|SegmentAnything_2023, p.2]])
> SA-1B, collected fully automatically using the final stage of our data engine, has *400× more masks than any existing segmentation dataset* [66, 44, 117, 60], and as we verify extensively, the masks are of high quality and diversity.
### responsible AI
> ([[SegmentAnything_2023.pdf#page=2&selection=176,50,179,41|SegmentAnything_2023, p.2]])
>  Images in SA-1B span a *geographically and economically diverse set of countries* and we found that SAM performs similarly across different groups of people.

![[SegmentAnything_2023.pdf#page=3&rect=31,131,563,750|SegmentAnything_2023, p.3]]
# task
> ([[SegmentAnything_2023.pdf#page=4&selection=19,13,56,56|SegmentAnything_2023, p.4]])
> The promptable segmentation task, then, is to return a **valid segmentation mask** given any prompt. The requirement of a “valid” mask simply means that even when a prompt is ambiguous and could refer to multiple objects (e.g., recall the shirt vs. person example, and see Fig. 3), *the output should be a reasonable mask for at least one of those objects*. This requirement is similar to expecting a language model to output a coherent response to an ambiguous prompt. We choose this task because it leads to a natural pre-training algorithm and a general method for zero-shot transfer to downstream segmentation tasks via prompting.

## pre-training
> ([[SegmentAnything_2023.pdf#page=4&selection=68,0,82,0|SegmentAnything_2023, p.4]])
> We adapt this method from **interactive segmentation** [109, 70], although unlike interactive segmentation **whose aim is to eventually predict a valid mask after enough user input**, *our aim is to always predict a valid mask for any prompt even when the prompt is ambiguous*

## zero-shot transfer
> ([[SegmentAnything_2023.pdf#page=4&selection=92,20,97,42|SegmentAnything_2023, p.4]])
> the ability to respond appropriately to any prompt at inference time, and thus downstream tasks can be solved by engineering appropriate prompts. For example, *if one has a bounding box detector for cats, cat instance segmentation can be solved by providing the detector’s box output* as a prompt to our model.

> ([[SegmentAnything_2023.pdf#page=4&selection=116,0,127,0|SegmentAnything_2023, p.4]])
> a broadly capable model that *can adapt to many* **(though not all)** *existing and new segmentation tasks via prompt engineering*.

> ([[SegmentAnything_2023.pdf#page=4&selection=140,41,147,18|SegmentAnything_2023, p.4]])
>  An important distinction in our work is that a model trained for *promptable segmentation can perform a new, different task at inference time* by acting as a component in a larger system

![[SegmentAnything_2023.pdf#page=5&rect=45,619,550,725|SegmentAnything_2023, p.5]]
# segment anything model
[[SegmentAnything_2023.pdf#page=5&selection=34,3,34,25|SegmentAnything_2023, p.5]]
## image encoder
- masked auto-encoder (MAE) vision transformer (ViT)
## prompt encoder
- sparse
	- points, boxes
		- positional embeddings
	- text
		- CLIP
- dense
	- masks
		- convolution
## mask decoder
- transformer decoder block
	- prompt self-attention
	- cross-attention (image <-> prompt)
	- upsampling
- dynamic mask prediction head
	- MLP
	- dynamic linear classifier


## losses and training
linear combination of focal loss and dice loss
# data engine
## stages
- assisted manual
	- annotators label "stuff" or "things", without labels
	- SAM runs live to segment based on annotations
	- annotators can use precise brush and erase tools
	- the model was retrained several times during this phase, making the assistance better and making annotators more efficient
- semi-automatic
	- confident masks were pre-made, then annotators were prompted to add new ones
- fully automatic
	- prompted with 32x32 grid [[SegmentAnything_2023.pdf#page=6&selection=59,0,61,2|SegmentAnything_2023, p.6]]
	- determined confident masks based on IoU
	- determined stable masks by thresholding at 0.5 +/- $\delta$
# dataset
- high resolution images (3300x4950 vs 480x640 for coco)
- masks are enirely automatically generated
- 