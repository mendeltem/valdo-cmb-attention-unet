# CMB-Detektion auf T2* mit einem 3D-Attention-U-Net (VALDO Task 2)

Zaehlt zerebrale Mikroblutungen (CMB) auf T2*-gewichteten Aufnahmen. Eingang ist
T2* plus ein FRST-Kanal; das Netz ist ein 3D-Attention-U-Net; nachbearbeitet wird
ueber Zusammenhangskomponenten (26er-Nachbarschaft) und einen T1-basierten Filter
gegen Fehlalarme.

**Status: Aufbau steht, Ergebnisse folgen.** Die Tabellen unten werden aus den
Laeufen gefuellt und sind bis dahin leer.

## Warum Zaehlen und nicht Dice

Eine Mikroblutung ist wenige Voxel gross. Ein Dice ueber ein ganzes Volumen sagt
darueber fast nichts -- ein Netz kann bei hohem Dice die Haelfte der Blutungen
uebersehen. Gemessen wird deshalb je Blutung:

Vorhersage und Referenz werden mit 26er-Nachbarschaft in Komponenten zerlegt
(`ndimage.label(m, structure=np.ones((3,3,3)))`); eine Komponente ist eine
Blutung. Dann gilt

- **TP** = Referenzkomponente, die von mindestens einer Vorhersagekomponente beruehrt wird
- **FN** = Referenzkomponente ohne Beruehrung
- **FP** = Vorhersagekomponente, die keine Referenzkomponente beruehrt
- mehrere Vorhersagen auf **einer** Referenz zaehlen 1 TP und 0 FP
- **F1** = 2TP / (2TP + FP + FN), gepoolt ueber alle Faelle

Dazu FP je Fall und der Zaehlfehler |n_vorhersage - n_referenz|. TN gibt es nur
auf Fallebene. Implementierung: `code/metrik.py` (mit `--selbsttest`).

## Datensatz

VALDO Challenge Task 2 (MICCAI 2021), 72 Faelle, 236 CMB. Drei Kohorten mit sehr
verschiedener Aufnahmegeometrie -- das ist fuer ein 3D-Netz der entscheidende
Punkt und wird im Aufbau ausdruecklich behandelt:

| Kohorte | n | CMB | pixdim (mm) | Anisotropie |
|---|---|---|---|---|
| 1 | 11 | 106 | 0,45 x 0,45 x 4,0 | 8,9:1 |
| 2 | 34 | 96 | 0,49 x 0,49 x 0,8 | 1,6:1 |
| 3 | 27 | 34 | 1,0 x 1,0 x 3,0 / 4,0 | 3-4:1 |

Ein fester Patch von 96x64x48 umspannt in Kohorte 1 physisch 192 mm in z, in
Kohorte 2 nur 38 mm. Ob ein gemeinsames Gitter (0,5 x 0,5 x 1,0 mm) besser ist als
natives Training, wird gemessen und nicht angenommen.

T1, T2, T2* und die CMB-Maske liegen bei 72/72 Faellen auf identischem Gitter
(shape und affine geprueft), die Bilder sind bereits hirnmaskiert. Daraus folgt
zweierlei: eine Hirnextraktion ist hier **nicht** noetig, und der T1-basierte
Filter braucht **keine Registrierung**.

**Die Daten sind hier nicht enthalten** (CC-BY-NC-4.0 erlaubt keine
Weiterverbreitung) und werden direkt bei der VALDO-Challenge bezogen.

## Aufbau

1. **Vorverarbeitung** -- FSL FAST `-B` (Biaskorrektur), FRST-Kanal und Overpaint
   nach dem microbleednet-Verfahren.
2. **Split** -- ~20 % der Faelle als Validierungsset, stratifiziert nach Kohorte
   und Blutungslast (0 / 1-2 / >=3), mit gierigem Ausgleich der CMB-Summe. Der
   Rest wird 5-fach kreuzvalidiert.
3. **Hyperparametersuche** -- Optuna (TPE) mit Pruning auf einer Falte; die
   besten drei Konfigurationen werden anschliessend ueber alle fuenf Falten bestaetigt.
4. **Nachbearbeitung** -- Mindestgroesse in mm3 (nicht in Voxeln: die Kohorten
   haben stark verschiedene Voxelvolumina), dann ein Filter, der Kandidaten in
   Ventrikel/CSF verwirft (SynthSeg auf dem mitgelieferten T1).

**Hauptmass sind die gepoolten CV-Zahlen.** Das Validierungsset enthaelt etwa 47
Blutungen; ein daraus gelesener F1 traegt rund +-0,1 Unsicherheit und ist
Plausibilitaetspruefung, kein Hauptergebnis. Schwellen, Mindestgroessen und
Hyperparameter stammen ausschliesslich aus Trainingsfalten.

## Ergebnisse

Gepoolte 5-fach-CV (Trainingsteil):

| Verfahren | F1 | Praezision | Sensitivitaet | FP je Fall |
|---|---|---|---|---|
| (folgt) | | | | |

Validierungsset, einmalig gemessen:

| Verfahren | F1 | Praezision | Sensitivitaet | FP je Fall |
|---|---|---|---|---|
| (folgt) | | | | |

## Quellen

Welcher Schritt woher stammt:

| Schritt | Quelle |
|---|---|
| Datensatz | VALDO Challenge Task 2, MICCAI 2021 -- https://valdo.grand-challenge.org (CC-BY-NC-4.0) |
| FRST-Kanal, Overpaint (`inpaint_vessels`), Kandidat + Diskriminator | Sundaresan et al., *microbleednet* -- https://github.com/v-sundaresan/microbleednet |
| Fast Radial Symmetry Transform (Grundlage des FRST-Kanals) | Loy & Zelinsky, *Fast Radial Symmetry for Detecting Points of Interest*, IEEE TPAMI 2003 |
| Netzarchitektur | Oktay et al., *Attention U-Net: Learning Where to Look for the Pancreas*, 2018 |
| Umsetzung der Architektur, Patch-Sampling, Augmentierung | MONAI -- https://monai.io |
| Biaskorrektur (`fast -B`) | Zhang, Brady & Smith, *Segmentation of brain MR images through a hidden Markov random field model*, IEEE TMI 2001 (FSL FAST) |
| T1-Segmentierung fuer den FP-Filter | Billot et al., *SynthSeg*, Medical Image Analysis 2023 (FreeSurfer `mri_synthseg`) |
| Hyperparametersuche (TPE, Pruning) | Akiba et al., *Optuna*, KDD 2019 |
| Hirnextraktion | Isensee et al., *HD-BET*, Human Brain Mapping 2019 -- **in dieser Linie nicht verwendet**, da die VALDO-Bilder bereits maskiert sind; im Code fuer nicht-maskierte Kohorten vorgesehen |

## Umgebung

Python 3, PyTorch 2.6 (CUDA 12.4), MONAI 1.5.2, Optuna 5.0, nibabel, scipy,
FSL 6.x, FreeSurfer 8.2.
