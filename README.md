## Image Summarization

### Authors: Hanna Strohm, Nathan White, Joseph Mohr 

[Project Proposal](https://github.com/hstrohm/ImageSummarization/blob/master/CS766%20Proposal.pdf)

[Midterm Report](https://github.com/hstrohm/ImageSummarization/blob/master/CS766%20Midterm%20Report.pdf)

### Introduction
For our project we decided to preform image summaration, which we define as the process of automatically describing images. Verbally describing images and their contents is an easy task for humans, but not necessarily so for computers. Both object detection and recognition and the relationships of different objects within a scene can be complex when it comes to image processing. Machine Learning has provided a viable solution to the first portion of this task, but the second portion remains difficult. We want to design an algorithmic approach that will be able to summarize the contents of a given image. This includes recognizing and describing common objects and their relation to other objects within the scene. After this, details regarding the objects of focus will also be added to the description. These two points will be the basis for the scene summarization with the main focus being the objects and how they relate to others within the scene, while extraneous details such as color and size can be added later. One example of this is an image of a human on their cell phone. While it’s easy to detect both objects individually, we want to be able to correlate the human with the phone to mean that “the human is using/holding/looking at the phone.” 

![image](https://user-images.githubusercontent.com/54555630/80897965-b87e0780-8cc3-11ea-8cab-7a5f6ae948bd.png) <br/>
Image Example (resize before) "The bear is black." is the computer generated response.

### Our Process
In order to achieve our goal, we designed a system that would allow us to provide some textual descriptions of an input image. After supplying an input image, we perform multiple-object detection, getting boundary boxes, masks, and labels for each object. From here, the image is sent to two stages, a backdrop removal as well as an object property detector.
The backdrop removal process attempts to remove items that aren’t the focus of the image.
Using the remaining portions of the image, this is sent through a spatial relations detector. Which determines how two objects in an image are related in terms of their position in the image.
From the results of the spatial relation detector, the objects and their relationships are turned into sentences using NLP techniques. For the object properties functions, these were intended to include object color, shape, etc; but due to time constraints only object color was implemented. The results of the object property functions are generated into simple sentences and then combined with the results of our NLP outputs to produce a textual description of the input image.


### Files:
**Code:**
* grab_model.ipynb - old notebook...think it was just testing the COCO dataset
* maskrcnn.ipynb - does vision stuff - gets data to be used by nlpstuff to make sentences
* nlpstuff.ipynb - does NLP stuff - makes sentences

**JSON:**
* results.json, failed.json - old method to transfer data about one image to NLP notebook
* everything.json - transfers data about (currently) 200 images to NLP notebook
* final_short_results.json - final results from NLP notebook, shortened version
* final_long_results.json - unshortened version
