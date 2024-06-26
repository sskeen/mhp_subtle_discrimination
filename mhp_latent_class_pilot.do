
*--------------------------------------------------------------------------------------------------
*
*	Linguistic markers of subtle discrimination among mental healthcare professionals
*		mhp_latent_class_pilot.do
*		Simone J. Skeen (06-26-2024)
*
*--------------------------------------------------------------------------------------------------

* Preparation
 
cd "C:\Users\sskee\OneDrive\Documents\02_tulane\01_research\tu_ceai\mhp_subtle_discrimination\data\wave 1"
import excel d_analysis_pilot, firstrow case(lower)
describe

rename fit fitt

* Pilot LCA

		*** SJS 6/26: removing agnt: too sparse, doesn't tap mhp semantics ie. separate MoA

*gsem (refl just afrm fitt <- _cons), family(bernoulli) link(logit) lclass(C 3) nonrtolerance
*estat lcprob
*estat lcmean
*marginsplot, noci

* Class enumeration: fit indices per _k_ classes

quietly gsem (refl just afrm fitt agnt <- ), family(bernoulli) link(logit) lclass(rq_class 1) iter(1000) nonrtolerance 
estimates store one_C

quietly gsem (refl just afrm fitt agnt <- ), family(bernoulli) link(logit) lclass(rq_class 2) iter(1000) nonrtolerance 
estimates store two_C

quietly gsem (refl just afrm fitt agnt <- ), family(bernoulli) link(logit) lclass(rq_class 3) iter(1000) nonrtolerance 
estimates store three_C

estimates stats one_C two_C three_C

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
*drop cpr* predclass
drop maxpr

predict cpr*, classposteriorpr
list refl just afrm fitt cp* in 1/30

* Gen predclass var

		*** JN 11/9: gen var for predicted class membership; 
		
		*** JN 11/9: gen max posterior probability for each class, assign each ob to predicted class.

egen maxpr = rowmax(cpr*)
gen predclass = 1 if cpr1==maxpr
replace predclass = 2 if cpr2==maxpr
list cp* maxpr predclass in 1/30

* Class separation

table predclass, statistic(mean cpr1 cpr2)

tab predclass






