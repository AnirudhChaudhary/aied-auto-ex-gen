# aied-auto-ex-gen

# AI for Education Final Project - Fall 2024
### Group Members: Anirudh Chaudhary, Aneesh Durai, Naveen Nathan

## Research Question

## Problem Statement

## Approach

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
