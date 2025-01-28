
*---------------------------------------------------------------------------------------*
*																						*
*	Linguistic markers of subtle discrimination among mental healthcare professionals	*
*		mhp_latent_class_pilot.do														*
*		Simone J. Skeen (06-28-2024)													*
*																						*
*---------------------------------------------------------------------------------------*

* Preparation
 
cd "C:\Users\sskee\OneDrive\Documents\02_tulane\01_research\tu_ceai\mhp_subtle_discrimination\data\wave 1"
clear
import excel d_analysis_pilot, firstrow case(lower)
describe

rename fit fitt
rename prob prbl

* Tetrachorics

tetrachoric prbl refl just afrm fitt agnt t_bin i_bin, stats(rho se)

*-----------*
* Pilot LCA *
*-----------*

* Class enumeration: fit indices per _k_ classes

quietly gsem (refl just afrm fitt agnt <- ), family(bernoulli) link(logit) lclass(rq_class 1) iter(1000) nonrtolerance 
estimates store one_class

quietly gsem (refl just afrm fitt agnt <- ), family(bernoulli) link(logit) lclass(rq_class 2) iter(1000) nonrtolerance 
estimates store two_class

quietly gsem (refl just afrm fitt agnt <- ), family(bernoulli) link(logit) lclass(rq_class 3) iter(1000) nonrtolerance 
estimates store three_class

estimates stats one_class two_class three_class

* Optimal: _k_ = 2-class solution

quietly gsem (refl just afrm fitt agnt <- ), family(bernoulli) link(logit) lclass(rq_class 2) iter(1000) nonrtolerance startvalues(randompr, draws(20) seed(56))
estimates store rq_class2

* Latent class marginal probabilities 
 
estat lcprob

* Goodness of fit

estat lcgof 

* Class-specific conditional means

estat lcmean, nose
marginsplot, noci

* Posterior probability of class membership for each observation

*describe
*drop cpr* predclass maxpr


predict cpr*, classposteriorpr
list refl just afrm fitt agnt cp* in 1/30

*--------------------------------*
* Covariates of class membership *
*--------------------------------*

gsem (refl just afrm fitt agnt <-, logit) (rq_class <- t_bin), family(bernoulli) lclass(rq_class 2) iter(1000) nonrtolerance startvalues(randompr, draws(20) seed(56))
estimates store rq_class2_t

estat lcmean 
estat lcprob
*marginsplot, noci

*-------------------------*
* Assign class membership *
*-------------------------*

		*** JN 11/9: gen var for predicted class membership; 
		
		*** JN 11/9: gen max posterior probability for each class, assign each ob to predicted class.

egen maxpr = rowmax(cpr*)
gen predclass = 1 if cpr1==maxpr
replace predclass = 2 if cpr2==maxpr
list refl just afrm fitt agnt cp* maxpr predclass in 1/30

* Class separation

table predclass, statistic(mean cpr1 cpr2)

tab predclass

* Housekeeping
 
recode predclass 2=0
rename predclass engaged_cl

tab engaged_cl

*--------*
* Export *
*--------*

export excel using "d_causal_pilot.xlsx", firstrow(variables)

*--------------------------------*
* CausalNLP ATE robustness check *
*--------------------------------*

logit engaged_cl i.t_bin i_bin
margins t_bin, atmeans
marginsplot, noci

* for viz

list refl just afrm fitt agnt cp* maxpr engaged_cl in 1/30

*----------------------------------*
* End of mhp_latent_class_pilot.do *
*----------------------------------*
