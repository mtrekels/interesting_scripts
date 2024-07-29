
# setwd("D:/My Works/Bcube")
# sbs = read.csv("species_by_site_08July24.csv")
# sbt = read.csv("species_by_trait_08July24.csv")
options(warn = -1)

# install.packages("mvabund")
library(mvabund)
library(cluster)
# library(dplyr)


example(flower)

sbt <- flowerN # species by trait matrix
s <- dim(sbt)[1] # number of species

env <- data.frame(temp=c(2.3, 4.5, 6.8, 7.9), rain=c(200, 340, 240, 150)) # environment dataframe
n <- dim(env)[1] # number of sites

sbs<-as.data.frame(matrix(rpois(n*s,1.5),nrow=n, ncol=s)) # site-by-species counts

abund<- sbs[2,] # abundance of site 2

###############


fit=traitglm(sbs,env,sbt)
fit$fourth #print fourth corner terms

# r function for alien trait according to fit$fourth, as x*F*e (matrix product)


#####
# for loop for 10,000 alien species

xx <- as.data.frame(lapply(sbt,sample)) # generate possible aliens
x <- xx[1,] # an alien species

r <- function(alien_x,environment){
  1 # a function of x and env
}

# intrinsic growth of alien with x trait in the environment of site 2
r(x,env[2,])

alpha <- function(alien_x,sbt,sig=0.2){
  # a function of x and sbt
  # trait distance between x and trait by vegdist(,method="gower")
  yy <- as.matrix(cluster::daisy(rbind(alien_x,sbt),metric = "gower"))
  td <- yy[s+1,1:s] # alien-resident trait distance
  exp(-td^2/(2*sig^2))
}

# interaction strength between alien and resident species
alpha(x,sbt)

inv <- function(alien_x,environment,sbt,ab){
  r(alien_x,environment) - alpha(alien_x,sbt) %*% t(ab)
  } # invasiveness

# invasiveness of alien x in site 2
inv(x,env[2,],sbt,sbs[2,])

#####################
####################


#####
# for loop for 10,000 alien species

# Calculating invasibility of site 2

inb = list()
for(t in 1:1000){
  xx <- as.data.frame(lapply(sbt,sample)) # generate possible aliens
  x <- xx[1,] # an alien species
  inb[t] = inv(x,env[2,],sbt,sbs[2,])
}
invs <- do.call(rbind,inb)

# hist(invs)
# invasibility of site 2
sum(as.numeric(invs>0))/1000

# max invasiveness
max(invs)

