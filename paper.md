---
title: 'napari-ToMoDL: A Python package with a napari plugin for accelerated reconstruction of tomographic images'
tags:
authors:
  - name: Marcos Obando
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Germán Mato 
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Teresa Correia
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University, USA
   index: 1
 - name: Institution Name, Country c
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Recent advances in different tomographic methodologies have contributed to a broad range of applications and research areas, from x-ray imaging dealing with medical [@ginat2014advances] and industrial [@de2014industrial] applications to optical sectioning, which provides a mesoscopic framework to visualise translucent samples [@sharpe2004optical], just to name a few. Once data is acquired, we face several challenges: first, artifacts should be corrected in a preprocessing stage if needed, then  raw projections should be reconstructed via a mathematical algorithm, and, finally, the results should be visualized in a suitable way.

We present here napari-ToMoDL, a plugin of the napari viewer [@chiu2022napari]  that contains four main methods for tomographic reconstruction: filtered backprojection (FBP) [@kak2001principles], Two-step Iterative Shrinkage/Thresholding (TwIST) [@bioucas2007new], U-Net [@ronneberger2015u] and ToMoDL [@obando2023model],  being the last our recent introduced method for optical projection tomography reconstruction. Tne neural network based techniques have been trained in the PyTorch framework [@NEURIPS2019_9015] and they display excellent results when the reconstruction is performed with a very sparse set of projections [@obando2023model]. The plugin also offers the capability of axis alignment via variance maximization [@walls2005correction]. 

The input to the plugin is a stack of projections. The user only has to determine which is the rotation axis of the system (vertical or horizontal) and choose the reconstruction method. Additional options include:

- resizing
- manual or automatic center-of-rotation alignement
- clip to circle
- filtering
- full or partial volume reconstruction
- choice of use of CPU or GPU for the reconstruction.

napari-ToMoDL is integrally based on well-established open source software libraries such as NumPy [@harris2020array], Scipy [@virtanen2020scipy] and scikit-image [@scikit-image]. The neural network methods in the software are implemented in PyTorch [@NEURIPS2019_9015]. The computational burden that the Radon transform poses when applied iteratively is  overcome by using  TorchRadon [@ronchetti2020torchradon], a fast differentiable routine for computed tomography reconstruction developed as a PyTorch extension.


# Statement of need

The problem of reconstruction of tomographic images is crucial in several areas. For this purpose, several libraries have been introduced within the Python language to alleviate this burden, such as scikit-image [@scikit-image], the ASTRA toolbox [@van2016fast] and TorchRadon [@ronchetti2020torchradon]. Nevertheless, the usage of these tools requires specific knowledge of image processing techniques and they not form by themselves an accessible end efficient data analysis and visualization framework that can be used by experimentalists to handle raw tomographic data.

From the side of the visualizers, napari [@chiu2022napari], a fast and practical bioimaging visualiser for multidimensional data, has rapidly emerged as a hub for high performance applications spanning a broad range of imaging data, including microscopy, medical imaging, astronomical data, etc. Therefore there is a context with an extensive offer of tomographic reconstruction algorithms and a lack of software integration for image analysts that enables them to access to other complex tasks such as segmentation [@ronneberger2015u] and  tracking [@wu2016deep], 

The software presented here intends to close the gap between a wide variety of reconstruction techniques and a powerful visualization tool by introducing  a ready-to-use widget that offers state-of-the-art methods for tomographic reconstruction as well as a framework that will allow to include future methodologies.

# Methods and Workflow

The reconstruction methods implemented in the the packages are:

- **FBP** Filtered backprojection is a widely used method for tomographic reconstruction. The method involves filtering the data in the frequency domain and then backprojecting the filtered data onto the 3D volume. The filter used in FBP is typically a ramp filter, which amplifies high-frequency components of the data. FBP is computationally efficient and works well for simple geometries, such as parallel-beam tomography.
- **TwIST** (Two-step Iterative Shrinkage and Thresholding) is an iterative method for tomographic reconstruction, which involves iteratively solving a convex optimisation problem such as (1) using the shrinkage and thresholding ­ technique for each 2D slice. In this implementation, we chose to minimise the total variation norm as our regularising function. TwIST can handle a wide range of geometries and produces high-quality reconstructions. However, it is computationally expensive and requires careful tuning of ­ parameters [@correia2015accelerated].
- **Unet** is a deep learning architecture for tomographic reconstruction that uses a U-shaped network with skip ­connections [@ronneberger2015u] . The proposed network in [@davis2019convolutional] processes undersampled FBP reconstructions and outputs streak-free 2D images. The skip connections help preserve fine details in the reconstruction and the network can handle complex geometries and noisy data. While reconstruction times for this approach are short, making it suitable for real-time imaging, training a U-Net requires a large amount of data.
- **ToMODL** is a method that combines  iteration over a data consistency step and an image domain artefact removal step achieved by a convolutional neural network. The data consistenmcy step is implemented using the gradient conjugate algorithm and the artefact removal via a deep neural network  with shared weights across iterations.  As the forward model is explicitly accounted for, the number of network parameters to be learned is significantly reduced compared to direct inversion approaches, thereby providing better performance in training data constrained settings [@obando2023model].

In Fig. \autoref{fig:workflow}, a complete pipeline describing the usage of napari-tomodl is presented. Based on the single-channel raw data acquired by a parallel tomography use-case, this ordered stack of files undergoes the following steps in order to obtain ready-to-analyse reconstructions:

1. Load the raw image stack using napari file manager tab, obtaining a new raw 3D image layer.
2. Over the plugin's sliding window, select the target raw image layer and click over the "Select image layer" button.
3. Rotation axis can be aligned using automatic/manual methods. For the manual way, select an integer corresponding to the pixel shift respect to the rotation axis. A single slice reconstruction (as described in step ) is recommended for manual iteration over this step.
4. Use filtering for reconstructions, only applying for FBP methods, and size of the reconstructed 'xy' slice.
5. Choose between FBP (analytical), TwIST (iterative), U-Net or ToMoDL methods (deep learning).
6. Choose to reconstruct full volume or single slice, specifying slice index in this case.
7. Regarding the alignment of the loaded volume, choose rotation axis between vertical or horizontal.

Once these steps are completed, the 'Reconstruction' button allows for executing the desired specifications for image recovery from projections. In napari, outputs are written as image layers which can be analysed by other plugins and saved in different formats. One special feature that napari offers on top of 3D images is volume renderization, useful once a full volume is computed with the presented plugin. Normalization of intensity and contrast can be also applied to specific layers using napari's built-in tools in the top-left bar.



![Napari-tomodl usage pipeline, described step-by-step from a stack of raw projection acquisition to reconstruction of single specific slice or full volume.\label{fig:Workflow}](./napari-tomodl/figures/Figura1.pdf)

# Use cases

We present three parallel tomography use cases for the napari-tomodl plugin:

1. Optical projection tomography (OPT)
Projection data of wild-type zebrafish (Danio rerio) at 5 days post fertilisation were obtained  using 4x objective projections. Using a rotatory cylinder, transmitted projections images were acquired with an angle step of 0.5 degrees. The acquired projections had 700 × 700 pixels with a resolution of 1.3 μm per pixel [bassi2015optical]. These projections were resampled to have a resolution of 100 × 100 pixels in order to reduce the computational complexity.
of the training phase.
2. High resolution X-ray acquisition (XRAY)
Projection data from a foramnifera were obtained using 20 KeV X rays and a high resolution detector with 1024x1280 pixels (5 μm per pixel). A rotatory suppert was used to acquire 360 prjections with 1 degree interval. The projections were resampled to 256x320 to reduce computational complexity. The raw data was processed using phase contrast techniques to improve contrast [@Paganin2002]. 
3. Positron emission tomography (PET)
Whole-body PET/CT images were acquired using a Biograph mCT PET/CT scanner (Siemens, Healthcare GmbH, Erlangen, Germany) and were initiated approximately 60 min after intravenous tracer administration. Diagnostic CT scans of the neck, thorax, abdomen and pelvis (200 reference mAs; 120 kV) were acquired 90 sec after intravenous injection of a contrast agent (90–120 ml Ultravist 370, Bayer AG) (https://autopet.grand-challenge.org/Dataset/) .

In Fig. \autoref{fig:data} we show examples of the projections used for the reconstrction process and a veiew of the 3D volume otained using the plugin with the ToMoDL option.

![Examples of the applicaton of the plugin. Left panels: examples of projection data for the thrre cases: OPT, XRAY and PET. Right panels: view of the reconstructed 3D volume for the same cases.\label{fig:data}](./napari-tomodl/figures/Figura1.pdf)


# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
