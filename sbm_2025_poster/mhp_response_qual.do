
*-----------------------------------------------------------------------------------------*
*																						  *
*	Reflecting patients' identities yields more engaged clinical communications overall	  *
*		mhp_response_qual.do														          *
*		Simone J. Skeen (03-18-2024)													  *
*																						  *
*-----------------------------------------------------------------------------------------*

* prep

clear all
set more off

* install

foreach i in colrspace heatplot lgraph palettes ///
schemepack spider  {
	ssc install `i', replace
	}

* set scheme, font

set scheme white_tableau
graph set window fontface "Arial"

* import

cd "C:\Users\<my_wd>"
clear

import excel d_mhp_pilot, firstrow case(lower)

* inspect

d
browse

* housekeeping

drop index response prbl rationale notes prmr_outcome scnd_outcome

ren (emailpairid withinpatientid firstinpair transgender insurance) ///
	(EmailPairID WithinPatientID FirstInPair t_prob i_prob)

foreach i of varlist refl-agnt {
	replace `i' = 0 if `i' == .
}

d

save d_mhp_pilot, replace

////////////////////// *-----------------------------------------------------------------* //////////////////////
////////////////////// * 1:1 Merge, d_annotated_pilot : AnalysDatCovidAttCDC22genETHlags * //////////////////////
////////////////////// *-----------------------------------------------------------------* //////////////////////
	
* SJS annotated response data: sort

use d_mhp_pilot, clear
sort EmailPairID WithinPatientID FirstInPair

* PB et al. experimental data: sort, 1:1 merge

use AnalysDatCovidAttCDC22genETHlags, clear
sort EmailPairID WithinPatientID FirstInPair
merg 1:1 EmailPairID WithinPatientID FirstInPair using d_mhp_pilot

* inspect w/ _merge var

list EmailPairID WithinPatientID FirstInPair if _merge != 3, sep(0)

		*** SJS 8/19: s/b equal to 245; _confirmed_

******************************* visual inspection: clean merge ******************************* 

save sbm_abstract_prelim, replace

* data reduction

drop if _merge != 3

keep EmailPairID WithinPatientID Message Posoutcome state StateFips female White ///
	Black Hispanic Anxiety AskedAboutTrans TransAllied NBAllied TransSpecialty Transornb ///
	Depression Stressed nonbinary Transgender Daysent Weeksent text refl just afrm fitt ///
	agnt t_prob t_bin i_prob i_bin

* encode state

encode state, gen(state_num)
d


////////////////////// *---------------------------* //////////////////////
////////////////////// * Descriptives, exploration * //////////////////////
////////////////////// *---------------------------* //////////////////////

* signals, disclosures

tabm female White Black Hispanic Transgender nonbinary Transornb

*dis 227 + 149

* response quality subthemes

tabm refl just afrm fitt agnt

* tetrachorics

tetrachoric refl just afrm fitt agnt, stats(rho se)

* gen intersectional newvars for LG 3-step entry

gen black_trans = 0
replace black_trans = 1 if (Black == 1) & (Transgender == 1)

tabm black_trans Black Transgender
list black_trans Black Transgender, sep(0)

gen latine_trans = 0
replace latine_trans = 1 if (Hispanic == 1) & (Transgender == 1)

gen black_trans_f = 0
replace black_trans_f = 1 if (Black == 1) & (Transgender == 1) & (female == 1)

gen latine_trans_f = 0
replace latine_trans_f = 1 if (Hispanic == 1) & (Transgender == 1) & (female == 1)

gen black_nb = 0
replace black_nb = 1 if (Black == 1) & (nonbinary == 1)

gen latine_nb = 0
replace latine_nb = 1 if (Hispanic == 1) & (nonbinary == 1)

save sbm_abstract_analytic, replace

* SPSS format for LG

export spss sbm_abstract_analytic

////////////////////// *-----------* //////////////////////
////////////////////// * Pilot LCA * //////////////////////
////////////////////// *-----------* //////////////////////

* class enumeration: fit indices per _k_ classes

local m refl just afrm fitt agnt

quietly gsem (`m' <- ), family(bernoulli) link(logit) lclass(rq_class 1) iter(1000) nonrtolerance 
estimates store one_class

quietly gsem (`m' <- ), family(bernoulli) link(logit) lclass(rq_class 2) iter(1000) nonrtolerance 
estimates store two_class

quietly gsem (`m' <- ), family(bernoulli) link(logit) lclass(rq_class 3) iter(1000) nonrtolerance 
estimates store three_class

estimates stats one_class two_class three_class

* optimal: _k_ = 2-class solution

quietly gsem (refl just afrm fitt agnt <- ), family(bernoulli) link(logit) lclass(rq_class 2) iter(1000) ///
	nonrtolerance startvalues(randompr, draws(20) seed(56))
estimates store rq_class2

		*** SJS 8/21: _note_ 2-class solution replicated in LG

* latent class marginal probabilities 
 
estat lcprob

* goodness of fit

estat lcgof 

* item response probabilities

estat lcmean, nose
return list

matrix A = r(table)

* margins plot - prelim

*marginsplot, noci

////////////////////// *------------* //////////////////////
////////////////////// * Spiderplot * //////////////////////
////////////////////// *------------* //////////////////////

 * display estat lcmean matrix
	
matrix list A	
 
* manual input for spider

clear
input clss mrkr mk_mean
	1 1 .40431901 
	1 2 .06737419
	1 3 .22970396
	1 4 .1615969
	1 5 .000000001
	2 1 .000000001
	2 2 .058599
	2 3 .000000001
	2 4 .04261174
	2 5 .07210504

	end

label define clssl 1 "Engaged (27%)" 2 "Detached (73%)"
label define mrkrl 1 "Reflect" 2 "Justify" 3 "Affirm" 4 "Fit" 5 "Agent"

label values clss clssl
label values mrkr mrkrl

spider mk_mean, by(mrkr) over(clss) alpha(8) msym(none) ra(0(0.1)0.5) rot(45) smooth(0) sc(black) palette(tol vibrant) lw(0.4)

* posterior probability of class membership for each observation

*d
*drop cpr* predclass maxpr

predict cpr*, classposteriorpr
list refl just afrm fitt agnt cpr* in 1/30

////////////////////// *---------------------------------------------------* //////////////////////
////////////////////// * 1-step estimation: covariates of class membership * //////////////////////
////////////////////// *---------------------------------------------------* //////////////////////

******************************* Autocoder.code_custom_topics: t_bin = reflections of TGD identty, concerns *******************************

clear
import excel sbm_abstract_analytic_t_bin_validated, firstrow case(lower)

save sbm_abstract_analytic, replace


* N = 755: all valid replies

quietly gsem (refl just afrm fitt agnt <-, logit) (rq_class <- t_bin), family(bernoulli) lclass(rq_class 2) iter(1000) ///
	nonrtolerance startvalues(randompr, draws(20) seed(56))
estimates store rq_class2_t

estat lcmean 
estat lcprob

* class 2 = "engaged"; pred t_bin as covar of class 2 membership

margins, at(t_bin=(0(1)1)) predict(classpr class(2))
marginsplot, scheme(s1color) name(marg1, replace)

* restriction to TGD disclosures (_n_ = 227)

quietly gsem (refl just afrm fitt agnt <-, logit) (rq_class <- t_bin) if transgender == 1, family(bernoulli) ///
	lclass(rq_class 2) iter(1000) nonrtolerance startvalues(randompr, draws(20) seed(56))
estimates store rq_class2_t

estat lcmean 
estat lcprob

* class 1 (epitomized by refl) = "engaged" analog; pred t_bin as covar of class 1 membership

margins, at(t_bin=(0(1)1)) predict(classpr class(1))
marginsplot, scheme(s1color) name(marg1, replace)

////////////////////// *-------------------------* //////////////////////
////////////////////// * Assign class membership * //////////////////////
////////////////////// *-------------------------* //////////////////////

egen maxpr = rowmax(cpr*)
gen predclass = 1 if cpr1==maxpr
replace predclass = 2 if cpr2==maxpr
list refl just afrm fitt agnt cp* maxpr predclass in 1/30

* class separation

table predclass, statistic(mean cpr1 cpr2)
tab predclass

* housekeeping
 
recode predclass 2=0
rename predclass engaged_cl

tab engaged_cl

save sbm_abstract_analytic

////////////////////// *---------------------------------------------* //////////////////////
////////////////////// * Explore crude subconstruct-specific effects * //////////////////////
////////////////////// *---------------------------------------------* //////////////////////

* loop over unadjusted OLS models

foreach x of varlist Black Hispanic female nonbinary Transgender black_trans latine_trans black_trans_f latine_trans_f black_nb latine_nb {
	foreach y of varlist afrm refl fitt just agnt {
		reg `y' `x'
	}
}

* latine_trans_f -> just, b = .41, p = .023
* latine_trans_f -> agnt, b = .15, p = .010

reg agnt Posoutcome latine_trans_f

		*** SJS 8/21: doesn't appear to be proxying pilot outcomes...
		
list latine_trans_f just agnt text if latine_trans_f == 1 & (just == 1 | agnt == 1), sep(0) 

* export to inspect

export excel using sbm_abstract_analytic, firstrow(variables)

////////////////////// *------------------------------------------------* //////////////////////
////////////////////// * Confirm adjusted subconstruct-specific effects * //////////////////////
////////////////////// *------------------------------------------------* //////////////////////

d

* latine_trans_f -> agnt

* adjusted logits w/ AMEs

logit agnt latine_trans_f i.daysent i.weeksent anxiety depression stress, nolog or
margins, dydx(latine_trans_f)

* check against LPM 

reg agnt latine_trans_f i.daysent i.weeksent anxiety depression stress

* latine_trans_f -> just

* adjusted logits w/ AMEs

logit just latine_trans_f i.daysent i.weeksent anxiety depression stress, nolog or
margins, dydx(latine_trans_f)

* check against LPM 
reg just latine_trans_f i.daysent i.weeksent anxiety depression stress

*----------------------------*
* End of mhp_sbm_abstract.do *			
*----------------------------*
