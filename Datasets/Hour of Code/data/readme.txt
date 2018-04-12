Overview
--------
This is a dataset of aggregate user interaction data for logged-in users working on two challenges, Hoc4 and Hoc18 from December 2013 to March 2014. In that time Code.org gathered over 137 million partial solutions. Submissions are collected each time a logged in user executed their code. This instantiation of the data was generated for the paper Autonomously Generating Hints by Inferring Problem Solving Policies by Piech, C. Mehran S. Huang J and Guibas L. Contact piech@cs.stanford.edu if you have questions.

General Statistics:
Statistic		Hoc4		Hoc18
Students Studied	509,405 	263,569
Submissions		1,138,506	1,263,360
Unique Submissions	10,293		79,553
Pass Rate		97.8%		81.0%
(Piech et Al. Inferring Problem Solving Policies)


Dataset Description
-------------------
The data for each problem is split into different directories, as follows

Asts
The Abstract Syntax Trees of all the unique programs. Each file is an AST in json format where each node has a "block type", a nodeId and a list of children. The name of the file is the corresponding astId. AstIds are ordered by popularity: 0.json is the most common submission, followed by 1.json etc. The file asts/counts.txt has corresponding submission counts and asts/unitTestResults.txt has the code.org unit test scores.

Graphs
Graphs/roadMap.txt stores the edges of the legal move transitions between astIds as allowed by the code.org interface.

GroundTruth
A dataset gathered by Piech et Al to capture teacher knowledge of "if a student had just submitted astId X, which adjacent astId Y would I suggest they work towards."

Trajectories
Each file represents a series of asts (denoted using their astIds) that one or more students went through when solving the challenge. File names are the trajectoryIds. The file trajectories/counts.txt contains the number of students who generated each unique trajectory. The file trajectories/idMap.txt maps "secret" (eg annonymized) studentIds to their corresponding trajectories.

Interpolated
A dataset gathered by Piech et Al. We interpolate student trajectories over the roadMap graph, so that for each student we attempt to calculate the Maximum A Posteriori path of single block changes that each student went through. The file interpolated/idMap.txt contains the mapping between interpolated trajectory Ids and trajectory Ids.

NextProblem
The students who attempted and completed the subsequent challenges (Hoc5 and Hoc19 respectively). The file nextProblem/attemptSet.txt is the list of "secret" (eg annonymized) studentIds of users who tried the next problem. The file nextProblem/perfectSet.txt is the list of "secret" (eg annonymized) studentIds of users who successfully completed the next problem. 

Unseen
Some ASTs do not compile in the interface and are thus not captured. This dir contains a list ASTs that do not compile but are still relevant for understanding user transitions. 

All the best!
