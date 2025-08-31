# Underwater Object Detection via Channel Stabilization  

The complex marine environment exacerbates the challenges of object detection manifold. With the advent of the modern era, marine trash presents a growing danger to the aquatic ecosystem, and it has always been challenging to address this issue effectively. Therefore, there is a significant need to **precisely detect marine waste and locate it accurately** in challenging underwater surroundings.  

To ensure the safety of the marine environment, underwater object detection is a crucial tool to mitigate the harmful impact of waste. This project explains **image enhancement strategies** and presents experiments exploring the best detection backbones after applying these methods.  

---

## 🌊 Key Contributions  
- **Channel Stabilization Technique**  
  - A lightweight image enhancement procedure to reduce haze and color cast (dominance of blue) in underwater images.  
  - Improves visibility and contrast, especially for **multi-scale objects**.  

- **Integration with Detectron2**  
  - Evaluated multiple backbones (e.g., **RetinaNet**, **Faster R-CNN**) on the [TrashCan dataset](https://conservancy.umn.edu/handle/11299/216171).  
  - Applied **augmentation techniques** and a **sharpening filter** to highlight object boundaries for better recognition.  

- **Comparisons with Deformable Transformer (Deformable DETR)**  
  - Benchmarked our method against transformer-based detectors.  
  - Channel stabilization + augmentation on top of the best-performing backbone yielded consistent improvements.  

---

## 📈 Results  

- On **TrashCan 1.0 (Instance version)**:  
  - **+9.53% absolute improvement** in Average Precision (AP) for small-sized objects.  
  - **+7% absolute gain** in bounding box AP compared to baseline.  

- Demonstrated enhanced detection of **small-scale marine debris** in cluttered and low-visibility underwater scenes.  

---

## 🔧 Methodology  

### 1. Preprocessing with RGHS  
Preprocessing input images using the **Relative Global Histogram Stretching (RGHS)** method.  

![RGHS Example](<img width="243" alt="Screen Shot 2023-02-07 at 7 06 10 PM" src="https://user-images.githubusercontent.com/57188476/217282454-064a850b-547f-472f-a6be-3705045f3f07.png">)  
*(Screenshot: Preprocessing of the given input image using Relative Global Histogram model)*  

---

### 2. Channel Stabilization  
Application of **Channel Stabilization** to reduce the dominance of blue color underwater, improving classification and detection of submerged waste.  

![Channel Stabilization Example](insert-your-image-path-here)  
*(Screenshot: Application of channel stabilization method)*  

---

### 3. Detection using Detectron2  
We explored different backbones within Detectron2, including **RetinaNet** and **Faster R-CNN**.  

![Detectron2 Example](insert-your-image-path-here)  
*(Screenshot: Detectron2 backbone comparison results)*  

---

## 📂 Dataset  
We conducted experiments on the **TrashCan Dataset** (Instance and Material versions).  
- Public dataset link: [TrashCan Dataset](https://conservancy.umn.edu/handle/11299/216171)  

---

## 🛠️ Tools & Frameworks  
- [Detectron2](https://github.com/facebookresearch/detectron2)  
- [PyTorch](https://pytorch.org/)  
- Deformable DETR  

---

## 🚀 Future Work  
- Extend channel stabilization to work in **real-time underwater scenarios**.  
- Incorporate **domain adaptation techniques** to generalize across different aquatic environments.  
- Explore **multi-modal fusion** (e.g., RGB + Sonar/HSI) for more robust underwater detection.  

---

## 📜 Reference  
If you use this work, please cite:  

```bibtex
@misc{ali2023underwater,
  title={Underwater Object Detection via Channel Stabilization},
  author={Ali, Muhammad and collaborators},
  year={2023},
  howpublished={\url{https://github.com/aliman80/Underwater-Object-Detection-via-Channel-Stabilization}}
}
