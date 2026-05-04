# OceanLens - Etat de l'art Flow Matching, downscaling et contraintes physiques

Date de synthese: 2026-04-29.

Contexte local: le projet actif est `/homelocal/emboulaalam/OceanLens_git`. Les notes locales indiquent deux environnements de travail principaux: LIR (`/scratch/emboulaalam/OceanLens_git`) et ECMWF HPCF/Atos AC (`/ec/res4/scratch/fra0606/work/OceanLens_git`). Les chemins de donnees historiques pointent surtout vers LIR, tandis que les notes compute recentes indiquent aussi une migration/duplication possible sur ECMWF.

## 0. Position actuelle d'OceanLens

OceanLens est deja proche de la ligne SOTA pour la super-resolution oceanique conditionnelle:

- Branche deterministe: CNO `mu = E[HR | LR]`, utilise comme estimateur moyen.
- Branche generative: Flow Matching sur residu, avec U-Net conditionne par `mu`/LR/mask selon variante.
- Couplage: `minibatch_ot`, donc le bruit initial est apparie aux cibles par transport optimal dans le batch.
- Temps d'entrainement: variante `v4_s1_logit_t` avec `t_sampling: logit_normal`.
- Contrainte physique: variantes `recoarsen` qui penalise `coarsen(HR_pred) != LR_native`.
- Inference: solveur Euler explicite dans `OceanLensSystem.sample`; plusieurs scripts utilisaient/visaient Heun, mais le code central actuel sample avec `x = x + v * dt`.
- Calibration: `cond_dropout_p` existe dans la loss, mais la config de base met `0.0`; le CFG n'est donc pas encore pleinement exploite sauf variante specifique a verifier.

Conclusion courte: la fondation est bonne. Les priorites SOTA sont maintenant: (1) CFG propre, (2) Heun dans le sampler central, (3) conservation re-coarsening plus systematique, (4) diagnostics spectraux/ensemble, (5) stabilite amplitude si le modele grossit, puis seulement ensuite DiT/AFNO.

## 1. Flow Matching, stochastic interpolants et transport optimal

### Flow Matching for Generative Modeling - Lipman et al., ICLR 2023

Source: https://arxiv.org/abs/2210.02747

Idee principale: entrainer un champ de vitesse de CNF sans simuler l'ODE pendant l'entrainement. Le modele apprend une regression de vitesse le long de chemins probabilistes fixes entre une distribution source et les donnees. Le cas utile ici est le chemin lineaire:

```text
x_t = (1 - t) x_0 + t x_1
v_target = x_1 - x_0
```

Pourquoi c'est important pour OceanLens:

- C'est exactement la formulation de `src/oceanlens/losses/fm.py`.
- Pour une super-resolution conditionnelle, `x_1` ne doit pas forcement etre le HR brut; il vaut mieux modeliser le residu `HR - mu` ou `HR - LR_up`, comme dans CorrDiff/AFM/Delefosse.
- Le chemin lineaire reduit la courbure et permet moins de pas d'inference qu'une diffusion DDPM classique.

Limites:

- Le papier fondateur n'est pas specialise aux contraintes conditionnelles physiques.
- Si les couples `(condition, target)` sont mal alignes, l'OT non conditionnel peut apparier des choses incompatibles. C'est une source potentielle d'instabilite pour OceanLens si les batchs melangent des regions/saisons tres differentes.

Decision OceanLens:

- Garder FM lineaire comme socle.
- Garder une cible residuelle plutot que HR direct.
- Pour `minibatch_ot`, eviter des batchs trop heterogenes: stratifier par region/saison ou ajouter un cout OT conditionnel incluant `mu`/LR.

### Stochastic Interpolants - Albergo & Vanden-Eijnden, ICLR 2023

Source: https://arxiv.org/abs/2209.15571

Idee principale: construire des normalizing flows via interpolants stochastiques entre deux distributions. Le papier formalise le lien entre ODE, SDE, score et interpolants. Il justifie qu'on peut travailler au niveau du champ de probabilite/ODE sans passer par une diffusion lente.

Apport pour OceanLens:

- Justifie l'approche ODE deterministe pour generer des echantillons.
- Donne une base theorique pour injecter du bruit/stochasticite de maniere controlee si les ensembles sont sous-dispersifs.

Limite:

- Le bruit additionnel n'est pas une rustine gratuite: il doit etre coherent avec l'interpolant ou calibre par validation ensemble.

Decision OceanLens:

- Ne pas ajouter du bruit arbitraire a chaque step sans diagnostic.
- Si on veut stochastic churn ou SDE-like sampling, le tester comme variante d'inference avec CRPS, rank histograms, spread/skill, spectres.

### OT-CFM - Tong et al., TMLR 2024

Source: https://arxiv.org/abs/2302.00482

Idee principale: Conditional Flow Matching generalise les chemins de FM; OT-CFM utilise un appariement optimal transport dans le minibatch pour produire des trajectoires plus simples et plus droites.

Apport pour OceanLens:

- C'est la justification directe de `_match_minibatch_ot`.
- Des trajectoires plus droites reduisent le nombre de pas necessaires et stabilisent l'entrainement.

Probleme:

- L'OT minibatch est local au batch et peut faire de mauvais appariements si les cibles sont conditionnellement incompatibles.
- Le cout actuel OceanLens est calcule sur `x1`/mask, pas explicitement sur la condition.

Decision OceanLens:

- Garder `coupling: minibatch_ot`.
- Ajouter une variante "condition-aware OT": cout = distance residu + lambda distance `mu`/LR/region/time.
- Comparer avec `independent` sur les memes seeds pour verifier que l'OT apporte bien une baisse NFE/CRPS et pas seulement une loss plus basse.

### Conditional Variable Flow Matching - Generale et al., 2024

Source: https://arxiv.org/abs/2411.08314

Idee principale: etendre FM aux densites conditionnelles avec variables de condition continues et OT conditionnel amorti. Le papier signale le probleme central: l'impact de la condition sur la dynamique ne doit pas etre ignore.

Apport pour OceanLens:

- Confirme que l'OT/FM conditionnel est un sujet distinct du FM generique.
- Pertinent si on melange bassins, saisons, fronts, zones cotieres et open ocean dans les memes batchs.

Decision OceanLens:

- Si les generations sont bonnes localement mais incoherentes par region/saison, passer de l'OT minibatch simple a un OT conditionnel ou a des batchs homogenes.

## 2. Downscaling meteo/ocean: residu, conservation et calibration

### CorrDiff - Mardani et al., NVIDIA

Sources: https://arxiv.org/abs/2309.15214 et version journal https://www.nature.com/articles/s43247-025-02042-5

Idee principale: downscaling km-scale en deux etapes: un U-Net deterministe predit la moyenne, puis un modele diffusion correcteur genere le residu. Le modele est conditionne par les champs basse resolution et restaure les distributions, les spectres et les extremes mieux qu'un modele deterministic-only.

Architecture/entrainement:

- Mean estimator deterministe.
- Correcteur generatif sur residu.
- Tache 25 km vers 2 km sur Taiwan.
- Evaluation par MAE/CRPS, distributions, spectres et cas extremes.

Problemes resolus:

- Le downscaling direct HR tend a lisser les details.
- Les structures fines sont stochasticques: plusieurs HR plausibles pour un meme LR.

Apport pour OceanLens:

- La decomposition `HR = mu + residual` est le bon design.
- Les metriques doivent etre probabilistes et spectrales, pas seulement RMSE.
- La calibration reste difficile meme dans CorrDiff; il ne faut pas supposer que le generatif est bien disperse sans rank/spread diagnostics.

Decision OceanLens:

- Garder CNO -> FM residuel.
- Suivre `mu`, residu, spectres, CRPS, spread-skill, rank histograms par variable et par region.

### Delefosse, Charantonis et al. - Super-Resolving Coarse-Resolution Weather Forecasts With Flow Matching, 2026

Source: https://arxiv.org/abs/2604.00897

Idee principale: super-resolution generative par Flow Matching comme post-processing de forecasts coarse. Formulation residuelle pour preserver les grandes echelles; evaluation par re-coarsening, qualite HR, ensembles et spectres.

Apport pour OceanLens:

- C'est le papier le plus directement aligne avec ta "re-coarsening loss".
- La contrainte `coarsen(HR_pred) ~= LR` n'est pas un detail: c'est un critere central de coherence de design.

Decision OceanLens:

- Activer/reprendre `recoarsen` comme contrainte standard apres pretraining FM, puis en fine-tuning.
- Mesurer la conservation par variable: `thetao`, `so`, `zos`, `uo`, `vo`.
- Pour `thetao`, suivre explicitement le bilan thermique moyen par patch et par bassin.

### Adaptive Flow Matching / Stochastic Flow Matching - Fotiadis et al., NVIDIA, ICML 2025

Sources: https://proceedings.mlr.press/v267/fotiadis25a.html et https://arxiv.org/abs/2410.19814

Idee principale: au lieu de partir d'un prior gaussien pur, encoder l'entree basse resolution vers une distribution de base plus proche de la cible, puis utiliser Flow Matching pour ajouter les details stochasticques. Le bruit est scale adaptativement selon l'incertitude de l'encodeur.

Problemes vises:

- LR et HR ne sont pas parfaitement alignes car les PDE a resolutions differentes suivent des trajectoires differentes.
- Certains canaux sont plus deterministes que d'autres.
- Donnees limitees et risque d'overfit.

Apport pour OceanLens:

- Ton CNO `mu` joue deja le role d'encodeur/deterministic base.
- L'etape manquante est l'incertitude adaptative: bruit initial ou amplitude residuelle conditionnee par la confiance de `mu`.

Decision OceanLens:

- Variante AFM: remplacer `x0 = randn_like(x1)` par `x0 = sigma(mu, LR, mask) * eps` dans l'espace residuel, ou apprendre une carte `sigma`.
- Evaluer par variable: `zos` et vitesses peuvent avoir une stochasticite differente de `thetao`.

### PC-AFM - Debeire et al., 2026

Source: https://arxiv.org/abs/2604.03459

Idee principale: ajouter a AFM des contraintes physiques soft de conservation et utiliser ConFIG/gradient surgery pour eviter les conflits entre objectif generatif et objectif physique.

Apport pour OceanLens:

- La re-coarsening loss peut entrer en conflit avec la loss FM si elle force trop tot ou trop fort.
- Le warmup actuel (`warmup_epochs`) est une bonne intuition; gradient surgery serait l'etape suivante si les gradients se contredisent.

Decision OceanLens:

- Garder warmup re-coarsen.
- Logger separement FM loss et recoarsen loss.
- Si la qualite fine baisse quand la conservation monte, tester gradient surgery ou alternance de batches/objectifs.

### SerpentFlow - Keisler et al., 2026

Source: https://arxiv.org/abs/2601.01979

Idee principale: alignement non apparie par decomposition structure partagee / detail specifique. Pour la super-resolution, la basse frequence est la structure commune; la haute frequence est generee conditionnellement.

Apport pour OceanLens:

- Utile si tu veux exploiter des donnees non parfaitement appariees LR/HR ou des domaines differents.
- Donne une justification pour separer explicitement basses frequences et hautes frequences, pas seulement `HR - mu`.

Decision OceanLens:

- Pas prioritaire si les paires GLORYS coarse/HR sont propres.
- Prioritaire si on veut transferer vers d'autres produits oceanographiques ou domaines non apparies.

## 3. Guidance, variance et sous-dispersion

### Classifier-Free Guidance - Ho & Salimans, 2022

Source: https://arxiv.org/abs/2207.12598

Idee principale: entrainer le meme modele en conditionnel et inconditionnel via condition dropout, puis a l'inference extrapoler:

```text
v_guided = v_uncond + w * (v_cond - v_uncond)
```

Dans plusieurs implementations, `w=1` correspond au conditionnel pur; `w>1` amplifie la composante conditionnelle. Attention aux conventions de code.

Probleme resolu:

- Les modeles conditionnels peuvent ignorer le bruit et devenir trop lisses/sous-dispersifs.
- CFG augmente la fidelite conditionnelle, mais peut reduire la diversite si trop fort.

Etat OceanLens:

- `cond_dropout_p` existe.
- La config de base met `0.0`, donc pas de vrai apprentissage unconditionnel.
- Le sampler central ne calcule pas encore `v_uncond` + `v_cond`.

Decision OceanLens:

- Entrainer une variante avec `cond_dropout_p` entre `0.1` et `0.2`.
- Ajouter CFG dans `sample`: deux passes FM par step, condition zero et condition normale.
- Sweeper `w`: `1.0, 1.5, 2.0, 2.5, 3.0`.
- Choisir `w` par spread/skill, CRPS, spectres et biais re-coarsen, pas par image visuelle seule.

### Classifier guidance - Dhariwal & Nichol, 2021

Source: https://arxiv.org/abs/2105.05233

Idee principale: guider la generation avec le gradient d'un classifieur externe.

Apport pour OceanLens:

- Moins adapte que CFG car il faudrait un classifieur/regresseur physique externe.
- Peut inspirer une guidance physique par gradient de conservation, mais PCFM est plus propre.

Decision OceanLens:

- Ne pas implementer classifier guidance standard.
- Si besoin de contrainte forte a l'inference, regarder PCFM/projection plutot qu'un classifieur.

### Hamill 2001 - rank histograms

Source: https://doi.org/10.1175/1520-0493(2001)129%3C0550:IORHFV%3E2.0.CO;2

Idee principale: les rank histograms aident a diagnostiquer fiabilite, biais et dispersion des ensembles, mais peuvent etre mal interpretes.

Apport pour OceanLens:

- Un ensemble sous-dispersif donne souvent un U-shape, mais un U-shape peut aussi venir d'un biais conditionnel.
- Il faut coupler rank histograms avec bias, spread/skill et CRPS.

Decision OceanLens:

- Ajouter diagnostics ensemble par variable et region.
- Ne pas regler CFG uniquement pour "augmenter l'ecart-type"; verifier le biais et la conservation.

## 4. Echantillonnage temporel et solveurs

### On the Importance of Noise Scheduling - Chen, 2023

Source: https://arxiv.org/abs/2301.10972

Idee principale: le choix de schedule bruit/temps est crucial et depend de la resolution/tache. Les regimes tres proches bruit pur ou donnee pure ne sont pas toujours les plus informatifs.

Apport pour OceanLens:

- Justifie de ne pas tirer `t` uniformement par defaut.
- Le milieu de trajectoire contient souvent les structures qui emergent: fronts, gradients, filaments.

Decision OceanLens:

- Garder `t_sampling: logit_normal` comme variante forte.
- Comparer uniform vs logit-normal sur loss de validation, spectres et CRPS.

### SD3 / Rectified Flow Transformers - Esser et al., 2024

Source: https://arxiv.org/abs/2403.03206

Idee principale: Rectified Flow a grande echelle, timestep/noise sampling biaise vers les echelles perceptuellement utiles, architecture MM-DiT, QK-Norm pour stabilite, scaling law.

Apport pour OceanLens:

- Confirme le choix `logit_normal`.
- QK-Norm devient important si on remplace le U-Net par Transformer/DiT ou si attention bf16 instable.

Decision OceanLens:

- Court terme: garder U-Net + attention, ajouter QK-Norm seulement si attention instable.
- Moyen terme: tester DiT seulement apres avoir stabilise CFG/conservation/calibration.

### EDM - Karras et al., 2022

Source: https://arxiv.org/abs/2206.00364

Idee principale: separer clairement les choix de design diffusion: preconditioning, schedule, solver, stochasticity. Introduit/standardise un sampling efficace type Heun dans ce cadre.

Apport pour OceanLens:

- Le sampler central devrait utiliser Heun plutot qu'Euler si on veut 25-50 steps robustes.
- Les choix de solver sont aussi importants que l'architecture.

Decision OceanLens:

- Implementer Heun dans `OceanLensSystem.sample`.
- Comparer Euler 20/50 vs Heun 20/25/50 sur cout et qualite.

### EDM2 - Karras et al., 2024

Source: https://arxiv.org/abs/2312.02696

Idee principale: magnitude-preserving layers, normalisation des poids/activations et post-hoc EMA pour eviter la derive d'amplitude pendant l'entrainement.

Apport pour OceanLens:

- Si le U-Net FM grossit, les problemes d'amplitude peuvent se traduire par residus trop faibles, trop forts ou instables.
- Utile pour stabiliser bf16 et long training.

Decision OceanLens:

- Court terme: gradient clipping, EMA, suivi des normes de `v_pred`, residu et activations.
- Moyen terme: MP-conv/MP-attention si derive d'amplitude observee.

## 5. Architectures SOTA pertinentes

### DiT - Peebles & Xie, ICCV 2023

Source: https://arxiv.org/abs/2212.09748

Idee principale: remplacer le U-Net par un Transformer sur patches latents. Les performances scalent avec les GFLOPs/profondeur/tokens.

Avantage:

- Vision globale, utile pour grands bassins et teleconnexions.
- Meilleur scaling si beaucoup de donnees/compute.

Limites pour OceanLens:

- Plus couteux.
- Moins inductif localement qu'un U-Net/CNO pour fronts et cotes.
- Les masques ocean/terre et geometries irregulieres doivent etre traites proprement.

Decision OceanLens:

- Pas en premiere priorite.
- Tester un DiT seulement apres une baseline U-Net FM bien calibree.
- Si teste: patch embedding avec mask channel, positional encoding geographique, QK-Norm, bf16 stable.

### FourCastNet / AFNO - Pathak et al., 2022

Source: https://arxiv.org/abs/2202.11214

Idee principale: utiliser des Adaptive Fourier Neural Operators pour prevoir la meteo globale a haute resolution efficacement dans l'espace spectral.

Avantage:

- Excellente efficacite pour dynamiques globales et longues portees.
- Spectral bias utile pour champs geophysiques.

Limites pour OceanLens:

- Domaine ocean avec masques/cotes: Fourier pur peut creer artefacts aux discontinuites.
- Plus adapte au forecast global qu'au detail local pres des cotes.

Decision OceanLens:

- AFNO est interessant pour une branche globale ou un backbone de `mu`.
- Pour la super-resolution regionale/cotiere, CNO/U-Net avec masks reste plus pragmatique.

### FNO - Li et al., 2021

Source: https://arxiv.org/abs/2010.08895

Idee principale: neural operator dans l'espace de Fourier, apprend des maps entre fonctions et permet du zero-shot super-resolution dans certains PDE benchmarks.

Apport pour OceanLens:

- Bonne reference pour l'idee "operateur" plutot que simple image-to-image.
- Moins robuste que CNO pres des masques et grilles irregulieres si utilise naivement.

Decision OceanLens:

- Garder FNO comme baseline theorique, pas comme remplacement immediat.

### CNO - Raonic et al., NeurIPS 2023

Sources: https://arxiv.org/abs/2302.01178 et implementation https://github.com/camlab-ethz/ConvolutionalNeuralOperator

Idee principale: adapter les convolutions pour apprendre des operateurs continus de PDE de facon robuste, avec meilleure invariance a la resolution que des CNN standards.

Apport pour OceanLens:

- Justifie le choix CNO pour `mu`.
- Compatible avec une vision "operator learning" de la relation LR -> HR moyenne.

Decision OceanLens:

- Garder CNO comme mean estimator.
- Si `mu` a des biais regionaux, ameliorer CNO/masks avant d'augmenter la taille du FM.

### SPADE / GauGAN - Park et al., CVPR 2019

Source: https://arxiv.org/abs/1903.07291

Idee principale: injecter les cartes semantiques dans les normalisations via scale/bias spatialement adaptatifs; les normalisations classiques peuvent effacer l'information semantique.

Apport pour OceanLens:

- Les masques terre/ocean, distance a la cote, bathymetrie, glace, region peuvent moduler les normalisations du U-Net.
- Actuellement le mask est surtout concatene comme condition; SPADE serait plus fort pour que l'info geographique survive en profondeur.

Decision OceanLens:

- Ajouter une variante SPADE/FiLM spatial dans les ResBlocks, conditionnee par mask + bathymetrie + LR/mu basse frequence.
- Prioritaire si le modele cree des artefacts pres des cotes ou efface des details geographiques.

## 6. Contraintes physiques: soft, hard et fine-tuning

### Physics-Constrained Flow Matching - Utkarsh et al., 2025

Source: https://arxiv.org/abs/2506.04171

Idee principale: appliquer des corrections/projections pendant le sampling FM pour satisfaire des contraintes physiques dures, sans re-entrainer le modele.

Apport pour OceanLens:

- Alternative a la re-coarsening loss: imposer `coarsen(HR) = LR` a chaque step ou en fin de step.
- Utile si la conservation doit etre garantie, pas seulement penalisee.

Decision OceanLens:

- Court terme: soft re-coarsening loss.
- Moyen terme: projection hard apres chaque step Heun:

```text
HR <- HR + upsample(LR - coarsen(HR))
```

avec mask-aware pooling et correction par variable.

### Physics-Constrained Fine-Tuning - Tauberschmidt et al., 2025

Source: https://arxiv.org/abs/2508.09156

Idee principale: fine-tuner un modele FM pre-entraine avec des residus PDE faibles/differentiables pour ameliorer la coherence physique et les problemes inverses.

Apport pour OceanLens:

- Applicable si on formalise des residus physiques ocean: geostrophie approx, divergence, conservation chaleur/sel, coherence `uo/vo/zos`.

Decision OceanLens:

- Ne pas commencer par PDE residuals lourds.
- Commencer par contraintes robustes: re-coarsening, spectres, gradients, conservation de moyenne.
- Ajouter PDE weak-form seulement si on peut les valider scientifiquement.

## 7. Architecture recommandee pour OceanLens vNext

### Baseline robuste a consolider

```text
LR native -> interpolation/mask -> CNO mean estimator mu
target = HR - mu
FM U-Net predicts velocity on target residual
condition = [mu, LR_up, mask, optional bathymetry/distance-to-coast]
training = FM loss + re-coarsening loss warmup
coupling = minibatch OT, preferably condition-aware
t_sampling = logit_normal
sampling = Heun, 25-50 steps
optional = CFG with condition dropout
```

### Changements prioritaires

1. Sampler Heun central
   - Remplacer Euler dans `OceanLensSystem.sample`.
   - Garder `num_inference_steps` configurable.

2. CFG complet
   - Entrainer avec `cond_dropout_p=0.1-0.2`.
   - Implementer `cfg_weight` en inference.
   - Sweeper `w` avec ensemble metrics.

3. Re-coarsening standard
   - Garder warmup.
   - Logger conservation par variable.
   - Tester projection hard a l'inference en option.

4. Diagnostics ensemble
   - CRPS, spread/skill, rank histogram, bias.
   - Spectres 2D/radiaux sur HR, prediction, residual.
   - Metrics par region: open ocean, cote, Mediterranee, Arctic/ice si pertinent.

5. Conditioning geographique
   - Ajouter distance-to-coast/bathymetrie si disponible.
   - Tester SPADE/FiLM spatial si artefacts cotiers.

6. Stabilite et scaling
   - EMA, grad clipping, logs de normes.
   - QK-Norm/EDM2-like seulement si derive observee.

### Ce qui n'est pas prioritaire immediatement

- Remplacer U-Net par DiT avant d'avoir CFG/conservation/calibration.
- Ajouter stochastic churn sans protocole de calibration.
- Ajouter PDE weak residuals complexes sans validation physique claire.
- Utiliser AFNO/FNO comme generatif principal sur domaines cotiers masques.

## 8. Grille de lecture des problemes OceanLens

### Modele trop lisse / cartes grises

Hypotheses:

- `cond_dropout_p=0`, pas de CFG effectif.
- FM apprend une moyenne conditionnelle et ignore la stochasticite.
- Target residuel trop penalise ou re-coarsening trop fort trop tot.

Actions:

- Activer CFG training/inference.
- Verifier variance residuelle par variable.
- Sweeper `recoarsen.weight`.
- Comparer `x0` gaussien standard vs adaptive sigma.

### Derive basse frequence / bassin trop chaud ou froid

Hypotheses:

- FM invente une composante basse frequence dans le residu.
- `coarsen(HR_pred)` ne respecte pas LR.

Actions:

- Activer re-coarsening.
- Ajouter projection hard optionnelle a l'inference.
- Evaluer bilan thermique/salinite par patch.

### Bons scores RMSE mais mauvais spectres

Hypotheses:

- CNO domine, FM n'ajoute pas assez de hautes frequences.
- Sampler trop peu precis ou CFG absent.

Actions:

- Spectral loss optionnelle sur residu, ou simple diagnostic d'abord.
- CFG sweep.
- Heun 25/50.

### Artefacts pres des cotes / glace

Hypotheses:

- Mask concatene trop faible; normalisations effacent l'information geographique.
- Pooling/downsampling mask imparfait.

Actions:

- Ajouter distance-to-coast/bathymetrie.
- Tester SPADE/FiLM spatial.
- Verifier mask-aware losses et re-coarsening sur cellules partiellement ocean.

### Sous-dispersion ensemble

Hypotheses:

- Conditionnement trop fort sans dropout.
- Bruit initial inadapte.
- Selection de `w` CFG trop faible ou trop forte selon convention.

Actions:

- CFG avec dropout.
- Adaptive sigma facon AFM.
- Rank histograms + CRPS + spread/skill, pas seulement std cible.

## 9. References rapides

- Flow Matching, Lipman et al. 2023: https://arxiv.org/abs/2210.02747
- Stochastic Interpolants, Albergo & Vanden-Eijnden 2023: https://arxiv.org/abs/2209.15571
- OT-CFM, Tong et al. 2024: https://arxiv.org/abs/2302.00482
- CorrDiff, Mardani et al.: https://arxiv.org/abs/2309.15214
- CorrDiff journal version: https://www.nature.com/articles/s43247-025-02042-5
- Delefosse/Charantonis/Bereziat 2026: https://arxiv.org/abs/2604.00897
- Adaptive Flow Matching, Fotiadis et al. 2025: https://proceedings.mlr.press/v267/fotiadis25a.html
- Stochastic Flow Matching preprint: https://arxiv.org/abs/2410.19814
- PC-AFM, Debeire et al. 2026: https://arxiv.org/abs/2604.03459
- SerpentFlow, Keisler et al. 2026: https://arxiv.org/abs/2601.01979
- CFG, Ho & Salimans 2022: https://arxiv.org/abs/2207.12598
- Classifier guidance/DDPM improvements, Dhariwal & Nichol 2021: https://arxiv.org/abs/2105.05233
- Noise scheduling, Chen 2023: https://arxiv.org/abs/2301.10972
- EDM, Karras et al. 2022: https://arxiv.org/abs/2206.00364
- EDM2, Karras et al. 2024: https://arxiv.org/abs/2312.02696
- SD3 / Rectified Flow Transformers, Esser et al. 2024: https://arxiv.org/abs/2403.03206
- DiT, Peebles & Xie 2023: https://arxiv.org/abs/2212.09748
- FourCastNet / AFNO, Pathak et al. 2022: https://arxiv.org/abs/2202.11214
- FNO, Li et al. 2021: https://arxiv.org/abs/2010.08895
- CNO, Raonic et al. 2023: https://arxiv.org/abs/2302.01178
- CNO implementation: https://github.com/camlab-ethz/ConvolutionalNeuralOperator
- SPADE, Park et al. 2019: https://arxiv.org/abs/1903.07291
- PCFM hard constraints, Utkarsh et al. 2025: https://arxiv.org/abs/2506.04171
- Physics-Constrained Fine-Tuning, Tauberschmidt et al. 2025: https://arxiv.org/abs/2508.09156
- Rank histograms, Hamill 2001: https://doi.org/10.1175/1520-0493(2001)129%3C0550:IORHFV%3E2.0.CO;2
- TorchCFM reference implementation: https://github.com/atong01/conditional-flow-matching
- diff2flow reference: https://github.com/CompVis/diff2flow
