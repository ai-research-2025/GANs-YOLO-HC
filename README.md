# GANs and YOLOv11 for Automated Cochlear Hair Cell Detection

This repository contains the code and experiments for our paper:

**GANs and YOLOv11 for Automated Cochlear Hair Cell Detection**  
_Cole Krudwig, Sara Avila, Ariana Mondiri, Adya Dhuler, Samantha Philips, Ashlyn Viereck, Kaylee Van Handel, Steven Fernandes_

We propose a deep learning–based approach using Generative Adversarial Networks (GANs) to generate synthetic cochlear hair cell (HC) images and a YOLOv11 object detection model to automate the counting of row-specific inner and outer HCs.

---

## Demo

A live demo of this project can be found at:
https://huggingface.co/spaces/AI-RESEARCHER-2024/Detection-of-Cochlear-Hair-Cells-YOLOv11

![YOLOv11 Demo](images/image1.png)

---

## Usage

You can run this project directly in **Google Colab** with no local setup required.

1. Navigate to the notebooks in the [`/src`](./src) directory.
2. Click the "Open in Colab" badge or open the notebook manually in Colab.
3. Select **Runtime → Change runtime type** and set hardware accelerator to **GPU**.
4. Run all cells from top to bottom.

> All required packages are installed in the first few cells of each notebook.

---

## Citation

If you build on this code or compare to it, please cite:

```bibtex
@inproceedings{krudwig2024gans,
  title={GANs and YOLOv11 for Automated Cochlear Hair Cell Detection},
  author={Cole Krudwig and Sara Avila and Ariana Mondiri and Adya Dhuler and Samantha Philips and Ashlyn Viereck and Kaylee Van Handel and Steven Fernandes},
  booktitle={Proceedings of the Springer FICTA Conference (LNCS Style)},
  year={2025}
}
```

## Acknowledgements

Cochlear image data was provided by the Dr. Richard J. Bellucci Translational Hearing Center at Creighton University.
