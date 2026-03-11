# Image Vision Tool – AI Processing & Filters
## Site-ul este accesibil la adresa <https://imageedit.ro/>

O aplicație interactivă pentru procesarea imaginilor care combină algoritmi clasici de Computer Vision cu tehnici moderne de Inteligență Artificială (AI). Proiectul permite utilizatorilor să aplice filtre, să detecteze obiecte și să mărească rezoluția imaginilor într-o interfață intuitivă.

## Caracteristici Principale
### Procesare Avansată (AI)
Colorează AI: Transformă imaginile alb-negru în imagini color folosind rețele neuronale.
Detectează Obiecte: Identifică și marchează obiectele din imagine (folosind modele precum YOLO sau SSD).
Mărire Rezoluție (Upscaling): Crește claritatea și dimensiunea imaginii (2x, 4x) fără a pierde detalii semnificative.

## Filtre și Efecte Speciale
Canale de Culoare: Manipulare la nivel de pixel pentru canalele Roșu, Verde și Albastru.
Efecte Clasice: Grayscale, Sepia, Negativ, Vintage.
Filtre Digitale: Detectare margini (Canny/Sobel), Emboss, Ascuțire (Sharpening) și efect de Cartoon.
Blur Control: Control variabil al intensității pentru efectul de estompare.

## Utilități Interfață
Drag & Drop: Încărcare rapidă a imaginilor.
Filtre Cumulative: Aplicarea mai multor efecte succesiv.
Istoric Persistent: Posibilitatea de a urmări modificările efectuate.
Sistem de Debug: Monitorizarea sesiunii active în timp real.

## Tehnologii Utilizate
Limbaj: Python
Interfață Grafică: Tkinter
Procesare Imagine: OpenCV, Pillow
Deep Learning: PyTorch si TensorFlow (pentru funcțiile AI de colorare și detecție)
NumPy: Pentru manipularea matricială a pixelilor.
