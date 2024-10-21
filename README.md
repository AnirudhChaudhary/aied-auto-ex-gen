# aied-auto-ex-gen

# AI for Education Final Project - Fall 2024
### Group Members: Anirudh Chaudhary, Aneesh Durai, Naveen Nathan

## Problem Statement
Currently, generating problems for courses is not easy. It takes a great amount of time to creatively think of a new problem, align it to course content, and verify that there is a sound solution. If this could be automated to some degree, instructors would have more time to support the students.

## Research Question
1. How can we automate the question generation process for instructors?


## Approach
We propose a two step approach that should automate the entire pipeline for instructors. At the base level, the instructor just need to input the course textbook and optionally, some reference practice problems. The pipeline will handle: <br />
1. Parsing the input text and any embedded examples into practice and guiding examples<br />
2. Creating an intermediate representation for the content (this can be a summary or graph)<br />
3. Converting the intermediate representation into exercises that have solutions to them


## Architecture

## Repo Structure
|--> README.md <br />
|--> agent.py (Agent Definition)<br /> 
|--> main.py (Controls interface with user)<br />
|--> material-to-KR.py (Converts course material to KR)<br />
|--> KR-to-Ex.py (Converts a KR to exercises)<br />

## Related Works / Inspiration
[Chain of Density](https://arxiv.org/abs/2309.04269)


## Future Work
1. Handle multiple forms of input (LATEX vs Images vs Videos vs Audio?)
2. Create different types of representations in the material to KR step (Concept Graph? Bullet Format? Essay?)
3. Be able to convert different types of KR to practice problems
4. Incorporate different types of solvers

## TODO
1. Gather input text that we can send into the material-to-KR - @Naveen
2. Work on the agent class so that it can interface with the OpenAI API for model @Aneesh
3. Create Infrastructure for agents to interact with each other - @Anirudh


## Problems Encountered
- AzureOpenAI doesn't like it if there is a local file named azure.py because it messes up some imports
