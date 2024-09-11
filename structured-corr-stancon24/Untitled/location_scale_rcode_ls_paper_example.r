############################################################################

# code corresponding to the illustrative example in the paper:
#
# Viechtbauer, W., & López-López, J. A. (in press). Location-scale models for
# meta-analysis. Research Synthesis Methods.

############################################################################

# install necessary packages (only need to do this once)

#install.packages("metafor")
#install.packages("numDeriv")

# load metafor package

library(metafor)

############################################################################

dat <- dat.bangertdrowns2004
dat

# collapse subjects down to three main groups

dat$subject <- factor(ifelse(dat$subject %in% c("Algebra", "Calculus", "Math", "Math in Science", "Statistics"), "Math",
                      ifelse(dat$subject %in% c("Chemistry", "Comp Science and Chemistry", "Biology", "Earth Science", "Natural Resources", "Nursing", "Science"), "Sci", "Soc")))

# fit standard random-effects model model

res1 <- rma(yi, vi, data=dat, test="knha")
print(res1, digits=3)

# obtain the prediction interval based on this model

predict(res1, digits=2)

# the standard random-effects model is a special case of the location-scale
# model with just intercept terms for the location and scale parts

res2 <- rma(yi, vi, data=dat, mods = ~ 1, scale = ~ 1, test="knha")
print(res2, digits=3)
predict(res2, newscale=1, transf=exp, digits=3) # same value of tau^2
predict(res2, newmods=1, newscale=1, digits=2)  # same prediction interval

# draw forest plot (including the results from the location-scale model with 'subject' as predictor)

op <- par(mar=c(3,4,1,2), tck=-0.012, mgp=c(2,0.5,0))

res <- rma(yi, vi, data=dat, test="knha")

forest(res, xlim=c(-5,4), at=seq(-1.5,2,by=0.5), ylim=c(-6,51), cex=0.6,
       rows=48:1, psize=1, addfit=FALSE, header=TRUE, digits=2L,
       mlab="", slab=paste(dat$author, dat$year, sep=", "),
       ilab=cbind(as.character(dat$subject), dat$ni), ilab.xpos=c(-2.5,-1.8))
abline(h=0)
text(-2.5, res$k+2, "Subject", font=2, cex=0.6)
text(-1.8, res$k+2, "n",       font=2, cex=0.6)

pred <- predict(res)
addpoly(pred$pred, sei=pred$se, ci.lb=pred$ci.lb, ci.ub=pred$ci.ub, pi.lb=pred$pi.lb, pi.ub=pred$pi.ub,
        row=-1, cex=0.6, digits=2, width=5)
text(x=-5, y=-1, pos=4, cex=0.6, bquote(paste("Random-Effects Model: k = ",
          .(formatC(res$k, digits=0, format="f")),", ", hat(tau)^2, " = ",
          .(formatC(res$tau2, digits=3, format="f")))))

text(-5, -2.5, pos=4, cex=0.6, "Location-Scale Model:")

res <- rma(yi, vi, mods = ~ subject - 1, scale = ~ subject - 1, data=dat, test="knha")
pred <- predict(res, newmods=diag(3), newscale=diag(3))
addpoly(pred$pred, sei=pred$se, ci.lb=pred$ci.lb, ci.ub=pred$ci.ub, pi.lb=pred$pi.lb, pi.ub=pred$pi.ub,
        row=c(-4,-5.5,-7), cex=0.6, digits=2, width=5)

tau2.subj <- predict(res, newscale=diag(3), transf=exp)
k.subj <- table(dat$subject)

text(x=-5, y=-4.0, pos=4, cex=0.6, bquote(paste("- Math Studies: k = ",
                                             .(formatC(k.subj[1], digits=0, format="f")),", ", hat(tau)^2, " = ",
                                             .(formatC(tau2.subj$pred[1], digits=3, format="f")))))

text(x=-5, y=-5.5, pos=4, cex=0.6, bquote(paste("- Science Studies: k = ",
                                             .(formatC(k.subj[2], digits=0, format="f")),", ", hat(tau)^2, " = ",
                                             .(formatC(tau2.subj$pred[2], digits=3, format="f")))))

text(x=-5, y=-7.0, pos=4, cex=0.6, bquote(paste("- Social Science Studies: k = ",
                                             .(formatC(k.subj[3], digits=0, format="f")),", ", hat(tau)^2, " = ",
                                             .(formatC(tau2.subj$pred[3], digits=3, format="f")))))

par(op)

# fit location-scale model with sample size as predictor

dat$ni100 <- dat$ni/100
res3 <- rma(yi, vi, mods = ~ ni100, scale = ~ ni100, data=dat, test="knha")
print(res3, digits=3)

# predicted tau^2 for 'small' versus 'large' studies

round(mean(dat$ni[dat$ni < 50]))
round(mean(dat$ni[dat$ni > 50]))
preds <- predict(res3, newscale=c(36,156)/100, transf=exp, digits=3)
preds
round(preds$pred[1] / preds$pred[2], 2) # ratio of tau^2 values

# bubble plots

op <- par(mfrow=c(2,1), mar=c(5,5,2,1))

ns100 <- seq(0,6,length=601)
preds <- predict(res3, newmods=ns100, newscale=ns100)

regplot(res3, pred=preds, xvals=ns100, pi=TRUE,
        xaxt="n", xlim=c(0,5.5), ylim=c(-1,1.5), las=1, bty="l",
        xlab="Sample Size", main="a) Sample Size and Location", digits=1,
        legend=TRUE)
axis(side=1, at=0:6, labels=seq(0,600,by=100))

hi <- hatvalues(res3)
tau2i <- pmax(0, resid(res3)^2 / (1-hi) - dat$vi)
wi <- sqrt(weights(res3))
size <- 0.5 + 3 * (wi - min(wi))/(max(wi) - min(wi))

plot(NA, NA, xlim=c(0,5.5), ylim=c(-0.13,1.3), xlab="Sample Size", xaxt="n",
     ylab=expression("Estimate of " * tau^2), las=1, bty="l", main="b) Sample Size and Scale")
axis(side=1, at=0:6, labels=seq(0,600,by=100))
pred <- predict(res3, newscale=ns100, transf=exp)
polygon(c(ns100, rev(ns100)), c(pred$ci.lb, rev(pred$ci.ub)), border=NA, col="gray85")
lines(ns100, pred$ci.lb, lty="dashed")
lines(ns100, pred$ci.ub, lty="dashed")
lines(ns100, pred$pred, lwd=3)
points(dat$ni100, tau2i, pch=21, col="black", bg="darkgray", cex=size)
legend("topright", inset=.01, bg="white", pch=c(NA,NA,22), col=c(NA,NA,"black"), pt.bg=c(NA,NA,"gray85"),   lty=c("blank",NA,NA),            lwd=c(NA,NA,1), text.col="white", pt.cex=1.5, seg.len=3, legend=c("Studies", "Regression Line", "95% Confidence Interval"))
legend("topright", inset=.01, bg=NA,      pch=c(21,NA,NA), col="black",          pt.bg=c("darkgray",NA,NA), lty=c("blank","solid","dashed"), lwd=c(1,3,1),   text.col="black", pt.cex=1.5, seg.len=3, legend=c("Studies", "Regression Line", "95% Confidence Interval"))

par(op)

# fit location-scale model with subject type as predictor

res4 <- rma(yi, vi, mods = ~ subject, scale = ~ subject, data=dat, test="knha")
print(res4, digits=3)

# note: the large negative scale coefficient for subjectSoc (and the huge SE)
# is a consequence of the very low amount of heterogeneity in this category;
# as a result, the coefficient wants to drift towards negative infinity (so
# that tau^2 = 0 for this category) while the standard error explodes

predict(res4, newmods=c(0,0), newscale=c(0,0), digits=2) # predicted effect for math studies
predict(res4, newmods=c(1,0), newscale=c(1,0), digits=2) # predicted effect for science studies
predict(res4, newmods=c(0,1), newscale=c(0,1), digits=2) # predicted effect for socsci studies

anova(res4, X=c(1,0,0),  digits=2) # test of predicted effect for math studies (same as testing the intercept)
anova(res4, X=c(1,1,0),  digits=2) # test of predicted effect for science studies
anova(res4, X=c(1,0,1),  digits=2) # test of predicted effect for socsci studies

predict(res4, newscale=c(0,0), transf=exp, digits=3) # predicted tau^2 for math studies
predict(res4, newscale=c(1,0), transf=exp, digits=3) # predicted tau^2 for science studies
predict(res4, newscale=c(0,1), transf=exp, digits=3) # predicted tau^2 for socsci studies

# equivalence check: fit random-effects models in the three subgroups (same
# estimated/predicted effects and tau^2 values for the three types of studies)

rma(yi,vi, data=dat, digits=3, test="knha", subset=subject=="Math")
rma(yi,vi, data=dat, digits=3, test="knha", subset=subject=="Sci")
rma(yi,vi, data=dat, digits=3, test="knha", subset=subject=="Soc")

# note: CIs are not exactly identical (as noted in footnote 7 in the paper, in
# the location-scale model, the KH method involves a single scaling factor and
# the degrees of freedom are taken to be k-p-1; on the other hand, when fitting
# separate random-effects models within each level of the categorical moderator,
# separate scaling factors are calculated within each level and the degrees of
# freedom are k_l−1, where k_l denotes the number of studies within level l of
# the moderator)

# equivalence check: a single mixed-effects model with subject type as a
# moderator while allowing tau^2 to differ across the three subject types

rma.mv(yi, vi, mods = ~ subject - 1, random = ~ subject | id, struct="DIAG", data=dat, test="t")

# note: CIs are again not identical, since the KH method is not available for
# models fitted with the rma.mv() function (test="t" also uses t-distributions
# for the tests/CIs, but not the scaling factor to adjust the var-cov matrix of
# the fixed effects as in the KH method)

# fit location-scale model with both predictors

res5 <- rma(yi, vi, mods = ~ ni100 + subject, scale = ~ ni100 + subject, data=dat, test="knha")
print(res5, digits=3)

# profile likelihood plots for the scale coefficients in the model

op <- par(mfrow=c(2,2), mar=c(5,5,2,1))

profile(res5, alpha=1, sub1=TRUE, cline=TRUE, steps=50, cex=0.8, xlim=c(-8,0))
profile(res5, alpha=2, sub1=TRUE, cline=TRUE, steps=50, cex=0.8, xlim=c(-8,1))
profile(res5, alpha=3, sub1=TRUE, cline=TRUE, steps=50, cex=0.8, xlim=c(0,10))
profile(res5, alpha=4, sub1=TRUE, cline=TRUE, steps=50, cex=0.8, xlim=c(-10,10))

par(op)

# profile likelihood CIs for the scale coefficients

confint(res5, alpha=1, digits=3, xlim=c(-8,0))
confint(res5, alpha=2, digits=3, xlim=c(-8,1))
confint(res5, alpha=3, digits=3, xlim=c(0,10))
confint(res5, alpha=4, digits=3, xlim=c(-10,10))

# simultaneous tests of the coefficients related to subject type

anova(res5, btt=3:4, digits=2) # location part
anova(res5, att=3:4, digits=2) # scale part

# predictions for the location part

predict(res5, newmods=c( 50/100,0,0), newscale=c( 50/100,0,0), digits=2) # math, n = 50
predict(res5, newmods=c(100/100,0,0), newscale=c(100/100,0,0), digits=2) # math, n = 100
predict(res5, newmods=c(150/100,0,0), newscale=c(150/100,0,0), digits=2) # math, n = 150

# predictions for the scale part

predict(res5, newscale=c(1,0,0), transf=exp, digits=3) # math,           n = 100
predict(res5, newscale=c(1,0,1), transf=exp, digits=3) # social science, n = 100
predict(res5, newscale=c(1,1,0), transf=exp, digits=3) # science,        n = 100

# fit location-scale model with different predictors for each part

res6 <- rma(yi, vi, mods = ~ ni100, scale = ~ subject, data=dat, test="knha")
print(res6, digits=3)

# profile likelihood CIs for the scale coefficients

confint(res6, alpha=1, digits=3, xlim=c(-12,0))
confint(res6, alpha=2, digits=3, xlim=c(0,10))
confint(res6, alpha=3, digits=3, xlim=c(-10,10))

# likelihood ratio test comparing model res6 with model res5

res5.ml <- update(res5, method="ML")
res6.ml <- update(res6, method="ML")
anova(res5.ml, res6.ml)

# refit all models with ML estimation

res1.ml <- update(res1, method="ML")
res2.ml <- update(res2, method="ML")
res3.ml <- update(res3, method="ML")
res4.ml <- update(res4, method="ML")
res5.ml <- update(res5, method="ML")
res6.ml <- update(res6, method="ML")

# table with IC values

fit.reml <- fitstats(res2, res3, res4, res5, res6)
fit.ml   <- fitstats(res2.ml, res3.ml, res4.ml, res5.ml, res6.ml)

tab <- data.frame(rbind(t(fit.ml[-2,]), t(fit.reml[-2,])))
tab

############################################################################
