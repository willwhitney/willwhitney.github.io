---
layout: post
title: "Computational power versus measured intelligence"
author: "Will Whitney"
tags: wild-speculation
---

After reading [Where The Falling Einstein Meets The Rising Mouse](http://slatestarcodex.com/2017/08/02/where-the-falling-einstein-meets-the-rising-mouse/) over on Slate Star Codex, I think that some of our confusion could be cleared up by defining intelligence more precisely. You should go read that post; Scott is a much better writer than I am, and does a great job of framing the issue. But as a reminder here's the basic dilemma, in his words:

> That is, we naturally think there’s a pretty big intellectual difference between mice and chimps, and a pretty big intellectual difference between normal people and Einstein, and implicitly treat these as about equal in degree. But in any objective terms we choose – amount of evolutionary work it took to generate the difference, number of neurons, measurable difference in brain structure, performance on various tasks, etc – the gap between mice and chimps is immense, and the difference between an average Joe and Einstein trivial in comparison. So we should be wary of timelines where AI reaches mouse level in 2020, chimp level in 2030, Joe-level in 2040, and Einstein level in 2050. If AI reaches the mouse level in 2020 and chimp level in 2030, for all we know it could reach Joe level on January 1st, 2040 and Einstein level on January 2nd of the same year. This would be pretty disorienting and (if the AI is poorly aligned) dangerous.  

However...

> In field after field, computers have taken decades to go from the mediocre-human level to the genius-human level.  

I think there's a really important source of confusion running through the post, and it's about how we measure intelligence.


## Two definitions of intelligence
There are two different definitions of intelligence which we tend to conflate, and they're not always linearly related:

* The number of computations a system can perform in a given amount of time, i.e. 1 [GFLOPS](https://en.wikipedia.org/wiki/FLOPS) vs. 1 TFLOPS.
* The set of problems a system can solve, i.e. "Play chess at the level of an average human" vs. "Play chess at the level of a grandmaster".

The field of computational complexity examines the relationship between these two quantities.


## What we talk about when we talk about intelligence
The way we reason from first principles about the intelligence of a system corresponds to the computations/time definition. How big is its brain? How quickly does computing improve (à la Moore's Law)?

On the other hand, when we discuss the _empirical_ intelligence of a system, we measure based on the "problems solved" definition. Ravens can solve multi-step puzzles. AlphaGo can beat the best human player.

With that in mind, what if the kind of empirical intelligence we care about increases less than linearly with computation? What if, for example, the ability to solve meaningful problems in the world increased with the log of computation instead?

Perhaps then the two AI progress plots from Scott's post are just talking at cross (ha-ha) purposes.

![](/assets/img/intelligence-and-computation/Slice.png)

In that case, it kind of makes sense that a lot of (computationally very different) systems fall within the human range of performance.

![](/assets/img/intelligence-and-computation/IMG_B23345B4CEC7-1.jpeg)

Under our new log-computation model of intelligence, the human range of computing power could fill half the scale of the natural world, even though the measured performance of humans (on the "Intelligence" axis) is only a little bit greater than the performance of chimps.

This is counterintuitive because our intuition is using the linear model. We assume that since humans have an EQ that's 3x that of chimps, there must be a big set of things that humans can do that chimps can't. On the log scale, though, our performance could be only fractionally better than the chimps.


## Does sublinear performance improvement make sense?
Let's look at that plot of computer chess performance over time, this time keeping Moore's Law in mind. Since computing doubles every 18 months, the time axis is equivalent to a log-scale computing axis.

![](/assets/img/intelligence-and-computation/ED85647E-99CE-4F64-9C5A-416F79262CD7.png)

That seems to imply that in a measure of intelligence we care about, the relationship between computation and intelligence is strongly sublinear.

Why should it take such a stupendously gigantic increase in computation to achieve such a subjectively small improvement in performance? It might be because very few problems can be solved in linear time or less. It's worth taking a look through the [table of time complexities](https://en.wikipedia.org/wiki/Time_complexity#Table_of_common_time_complexities) on Wikipedia to get a sense of just how trivial a problem has to be to solve it in low polynomial time.


### Asymptotic running times are really important

Perhaps to play chess perfectly you have to consider every possible trajectory of moves through the game. Then if the game lasts $$n$$ steps, and there are $k$ moves possible at each step, you have to consider $k^n$ ways that the game could go. Say your system currently has enough computing power to perfectly model chess games that are $n=10$ moves long. If you make your system $k$ times more powerful, it can still only play chess games that are 11 moves long. And to solve a 20-step game, you need $k^{10}$ times more computing power. That means that even if there were only two move options per turn, your system would need to be ~1000x more powerful in order to get 2x better performance.

Since the real branching factor of chess [is about 35](https://en.wikipedia.org/wiki/Branching_factor), going from 10-step to 20-step prediction would actually take 2,000,000,000,000,000 times as much computation.

Of course, exponential time is just about the worst-case scenario (though there are many [real problems](https://en.m.wikipedia.org/wiki/Travelling_salesman_problem) we care about that are NP-hard!). What if instead chess was ridiculously easy to solve perfectly, requiring only $O(n^2)$ time? To go from solving a 10-step chess game to an 11-step one, you need ~20% more processing power. To solve a 20-step game instead, you need 4x as much power. Instead of being a log plot we have a square root plot; it's a lot better than log, but you're still getting nowhere fast.

![](/assets/img/intelligence-and-computation/bokeh_plot%20(18).png)

In other words, we should be absolutely shocked if real-world performance actually did increase linearly with computation.


## The world doesn't want things to be smart
Performance increases sublinearly with computing power. This is such an ordinary fact that we lose sight of how _fascinating_ it is. We live in a world where at the most _fundamental_ level, it gets harder to make a system smarter the smarter it already is. If you have laptop computer that plays Go by brute force at the level of an eleven-year-old, and you want to keep adding processors until it beats Ke Jie, your computer will be bigger than the Sun.

This holds in an awful lot of situations. If someone builds a human-level AI and steals every computer in the world to turn it superhuman, it'll be able to do a lot more things at the same time, but it won't be able to predict the future or prove new theorems or break encryption. For most hard tasks, it will still be just about human-level.


## AI prognostication
All this together leaves us surprisingly close to where we started. If Moore's Law continues, we should expect a linear relationship between years and (a computational upper bound on) performance. That seems to imply that Eliezer's prediction is correct, and the time needed to go from a human-level system to a superhuman system will be shorter than the time between creating a chimp-level system and a human level-system. What it doesn't tell us is what the rate of progress — mouse to chimp, idiot to Einstein — might be.

However, we can look at the rate of progress in some historical examples and use that to guess just how much smarter Einstein is than the least smart human, measured in years of computing development.

In 1964 the Cray CDC 6600 was the fastest computer in the world [by an order of magnitude](https://en.wikipedia.org/wiki/History_of_supercomputing#Beginnings:_1950s_and_1960s). It was capable of 500,000 mathematical operations on floating-point numbers per second. A computer less powerful than the 6600 was already into the human range of performance in 1960. 37 years later, in 1997 Deep Blue managed to beat the best human player in the world; it used 11,000,000,000 floating-point computations per second to do so. That works out to a 22,000x speedup, or ~14.5 doublings, to get from beating a bad human at chess to beating the best human at chess.

![](/assets/img/intelligence-and-computation/BE98C14C-F425-4A8A-B2BD-998276E8197C.png)

In 1985, when the Go chart begins, the fastest computer in the world could perform 2.4 billion operations a second. AlphaGo (circa Lee Sedol) was powered by more than 50 Tensor Processing Units, each performing [92,000,000,000,000 matrix-multiply operations per second](https://cloud.google.com/blog/big-data/2017/05/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu) (92 TFLOPS). However, DeepMind was able to improve AlphaGo's efficiency such that it used only a single next-generation 180 TFLOPS TPU by the time of the match against Ke Jie. 2.4 GFLOPS in 1985 to 180 TFLOPS in 2017 represents a 75,000x speedup, or ~16 doublings. 31 years and 16 doublings isn't exactly the same as 37 years and 14.5 doublings, but it's surprisingly close.

I don't have much confidence in those numbers, but if we make an absolutely unfounded leap, we might guess that in other domains it will take 20-30 years for computers to go from outperforming the very worst humans at a task to outperforming the very best. Given that the DARPA Urban Challenge was in 2007, NASCAR drivers will be out of a job sometime a bit more than a decade from now.

Of course, the range of human performance is hugely different for different tasks. A world-class chess player might be ten times better at chess than an average human, while an Olympic sprinter is only twice as fast as a regular person. Bots are now as good at [writing](https://www.wired.com/2017/02/robots-wrote-this-story/) [news](https://www.recode.net/2017/7/7/15937436/google-news-media-robots-automate-writing-local-news-stories) [stories](https://www.theguardian.com/media/2016/apr/03/artificla-intelligence-robot-reporter-pulitzer-prize) as a very bad human (they still need a lot of help) but whether it will be 10 years or 50 before they can write a novel is very much unclear.

But if we think the gap between mice and men is a lot bigger than the gap between dumb humans and smart humans, we're going to be waiting a long long time for our mouse AI to grow up into Einstein.

<!-- ---








It's interesting how much time has passed between these different "human-level" milestones. Even two games as structurally similar as chess and Go required entirely different orders of magnitude of computing to solve, not to mention decades of research. To me this reinforces the point that humans are not general-purpose computers; if we were, the amount of computing needed to beat a Ke Jie in Go would be the same needed to beat Garry Kasparov in chess. This should influence the way people predict AI.











This reality gets obscured by the fact that the Moore's Law improvement of computing is exponential. If Einstein is twice as computationally powerful as the village idiot, who is in turn twice as powerful as the chimp, it would still only take 18 months to go from chimp to idiot and from idiot to Einstein.

In this context those plots of computer chess performance make more sense:


If we live in a world where intelligence





![](intelligence-and-computation/image.png)

In that case, if we look at just the "computation" axis,


![](intelligence-and-computation/2A2737FE-5451-449C-9315-FA4B9B211180.png)




![](intelligence-and-computation/FullSizeRender.jpg)


One source of uncertainty is that we've never created a system that was as good at a mouse, so we don't really have a clue what progress looks like. All the things we think of as intelligent are tasks like playing chess, which, to quote Scott again, "Mice can’t play chess (citation needed)." As far as things that mice actually do — like navigate a changing environment, pick things up, and raise children — our systems aren't even close.

From Scott's post:

> Stephen Hsu calculates that a certain kind of genetic engineering, carried to its logical conclusion, could create humans “a hundred standard deviations above average” in intelligence, ie IQ 1000 or so. This sounds absurd on the face of it, like a nutritional supplement so good at helping you grow big and strong that you ended up five light years tall, with a grip strength that could crush whole star systems. But if we assume he’s just straightforwardly right, and that Nature did something of about this level to chimps – then there might be enough space for the intra-human variation to be as big as the mouse-chimp-Joe variation.  

Given a log-scale relationship between the kind of intelligence where we're five light years tall and the kind of intelligence where a [baby chimp is about as smart as a human baby](http://www.smithsonianmag.com/smart-news/guy-simultaneously-raised-chimp-and-baby-exactly-same-way-see-what-would-happen-180952171/)



* Go getting superhuman in 20 years isn't a strong case; it also used 1000x power


With that in mind, what if the relationship between what we think of when we say " and what we think of as

relationship between what we think of as "improvement in AI", be that models or computing power, and what we think of as "ability of AI"


If the size of problems a system can solve was always proportional to how much computing power it has, this distinction wouldn't make any difference. A chimp being twice as smart as a mouse would always mean the same thing as a person being twice as smart as a chimp: twice as big a brain, twice as good at playing chess. But if it's not



Part of the problem is that we've never created a system that was as intelligent at a mouse, so we don't really have a clue what our progress looks like. As far as things that mice actually do — like navigate a changing environment, pick things up, and raise children — machines aren't even close to mouse-level. Since mice can't play chess, to say that a computer is human-level at chess rather than mouse-level is kind of tautological.

we don't get to include the leap from mouse to human in our estimate of the rate of chess progress.

All we really know is that it took almost forty years to go from a system that can play chess at all to a system that can beat every human at chess. -->
