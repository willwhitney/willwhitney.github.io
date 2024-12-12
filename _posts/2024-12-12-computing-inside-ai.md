---
title: Computing inside an AI
subtitle: What would it mean to treat AI as a tool instead of a person?
date: 2024-12-12
---

Since the launch of ChatGPT, there have been two explorations happening in parallel. 

The first direction is about technical capabilities. How big a model can we train? How well can it answer SAT questions? How efficiently can we serve it?

The second direction is about interaction design. How do we communicate with a model? How can we use it for useful work? What metaphor do we use to reason about it?

The first direction is widely followed and hugely invested in, and for good reason: progress on technical capabilities underlies every possible application. But the second is just as crucial to the field, and it has tremendous lingering unknowns. We are now only a couple of years into the large model age. What are the odds we've already figured out the best ways to use them?

I propose a new mode of interaction, where models play the role of computer (e.g. phone) applications: providing a graphical interface, interpreting user inputs, and updating their state. In this mode, instead of being an "agent" that uses a computer on behalf of the human, AI can provide a richer and more powerful computing environment for us to use.


## Metaphors for interaction

At the core of an interaction is a metaphor which guides a user's expectations about a system. The early days of computing took metaphors like "desktops" and "typewriters" and "spreadsheets" and "letters" and turned them into digital equivalents, which let a user reason about their behavior. You can leave something on your desktop and come back to it; you need an address to send a letter. As we developed a cultural knowledge of these devices, the need for these particular metaphors disappeared, and with them the skeumorphic interface designs that reinforced them. Like a trash can or a pencil, a computer is now a metaphor for itself.

The dominant metaphor for large models today is *model-as-person*. This is an effective metaphor because people have extensive capabilities that we have strong intuitions about. It implies that we can hold a conversation with a model and ask it questions; that the model can collaborate with us on a document or a piece of code; that we can give it a task to do on its own and it will come back when it's finished.

However, treating a model as a person profoundly limits how we think to interact with it. Human interactions are inherently slow and linear, limited by the bandwidth and turn-taking nature of speech. As we've all experienced, communicating complex ideas in conversation is hard and lossy. When we want precision, we turn to tools instead, using direct manipulation and high-bandwidth visual interfaces to make diagrams and write code and design CAD models. Because we conceptualize models as people, we use them via slow conversation, even though they're perfectly capable of accepting fast direct inputs and producing visual results. The metaphors we use constrain the experiences that we build, and model-as-person is keeping us from exploring the full potential of large models.[^notebooklm]

For many use cases, and especially for productive work, I believe that the future lies in a different metaphor: *model-as-computer*. 


## Using an AI as a computer

Under the model-as-computer metaphor, we will interact with large models according to the intuitions we have about computer applications (whether desktop, tablet, phone...). Note that this does not mean that the model will be a traditional app any more than the Windows desktop was a literal desk. "Computer application" will be a way for a model to represent itself to us. Instead of acting like a *person*, the model will act like a *computer*.

Acting like a computer means producing a graphical interface. In place of the charmingly teletype linear stream of text provided by ChatGPT, a model-as-computer system will generate something which resembles the interface of a modern application: buttons, sliders, tabs, images, plots, and all the rest. This addresses key limitations of the standard model-as-person chat interface:

- **Discoverability**. A good tool suggests uses for itself. When the only interface is an empty text box, the onus is on the user to figure out what to do and understand the boundaries of the system.[^voice-assistants] The Edit sidebar in Lightroom is a great way to learn about photo editing because it doesn't just tell you what this application *can* do to a photo, but what you might *want* to do. Similarly, a model-as-computer interface for DALL-E could surface new possibilities for your image generations. If you asked for an image in the style of a sketch, it could generate radio buttons for the drawing medium (pencil, marker, pastels, ...), a slider for the level of detail in the sketch, a toggle between color and B&W, and some illustrated buttons to select a perspective (2D, isomorphic, two-point perspective...).
- **Efficiency**. Direct manipulation is quicker than writing a request in words. To continue the Lightroom example, it would be unthinkable to edit a photo by telling a person which sliders to move and how much. You'd be there all day asking for a little lower exposure and a little higher vibrance, just to see what it would look like. In the model-as-computer metaphor, the model can create tools that let you communicate what you want more efficiently and thus get things done faster. In the DALL-E example, clicking through those options and dragging those sliders would let you explore a space of generated sketches in real time.

Unlike in a traditional application, this graphical interface is generated by the model on demand. This means that every part of the interface you see is relevant to what you're doing right now, including the specific contents of your work (the subject of this picture, the tone of this paragraph). It also means that if you would like more or different interface, you can just ask for it. You might ask DALL-E to produce some editable presets for its settings that are inspired by famous sketch artists. When you click on the Leonardo da Vinci preset, it sets the sliders for highly detailed perspective drawings in black ink. If you click Charles Schulz, it selects low detail technicolor 2D comics instead.


## A Protean bicycle of the mind

Model-as-person has a curious tendency to create distance between the user and the model, mirroring the communication gap between two people that can be narrowed but never quite closed. Because of the difficulty and expense of communicating in words, people tend to break up tasks amongst themselves into large chunks that are as independent as possible. Model-as-person interfaces follow this pattern: it's not worth telling a model to add a return statement to your function when it's quicker to write it yourself. With the overhead of communicating, model-as-person systems are most helpful when they can do an entire block of work on their own. They do things *for* you.

This stands in contrast to how we interact with computers or other tools. Tools produce visual feedback in real time and are controlled through direct manipulation. They have so little communication overhead that there is no need to spec out an independent block of work. It makes more sense to keep the human in the loop and directing the tool from moment to moment. Like seven league boots, tools let you go farther with each step, but you are still the one doing the work. They let *you* do things faster.

Consider the task of building a website using a large model. With today's interfaces, you might treat the model as a contractor or a collaborator. You would try to write down in words as much as possible about how you want the site to look, and what you want it to say, and what features you want it to have. The model would generate a first draft, you would run it, and then you would have some feedback. "Make the logo a little bigger", you would say, and "center that first hero image", and "there should be a login button in the header". To get things exactly the way you want, you will send a very long list of increasingly nitpicky requests.

An alternative model-as-computer interaction would look different: instead of *building the website*, the model would *generate an interface* for *you* to build it, where every user input to that interface queries the large model under the hood. Perhaps when you describe your needs it would make an interface with a sidebar and a preview window. At first the sidebar contains only a few layout sketches that you can choose as a starting point. You can click on each one, and the model writes the HTML for a web page using that layout and displays it in the preview window. Now that you have a page to work with, the sidebar gains additional options that affect the page globally, like font pairings and color schemes. The preview acts as a WYSIWYG editor, allowing you to grab elements and move them around, edit their contents, etc. Powering all of this is the model, which sees these user actions and rewrites the page to match the changes they make. Because the model can generate an interface to help the two of you communicate more efficiently, you get to exercise more control over the final product in less time.

Model-as-computer encourages us to think of the model as a tool to interact with in real time rather than a collaborator to give assignments to. Instead of replacing an intern or a tutor, it can be a sort of shape-shifting bicycle for the mind, one which is always custom-built exactly for you and the terrain you plan to cross.


## A new paradigm for computing?

Models that can generate interfaces on demand are a brand new frontier in computing. They may be a new paradigm entirely, with the way they short-circuit the existing application model. Giving end users the power to create and modify apps on the fly fundamentally changes the way we interact with computers. In place of a single static application built by a developer, a model will generate a bespoke application for the user and their immediate needs. In place of business logic implemented in code, the model will interpret the user's inputs and update the UI. It's even possible that this sort of *generative UI* will replace the operating system entirely, generating and managing interfaces and windows on the fly as needed.

At first, generative UI will be a toy, only really useful for creative exploration and a few other niche applications. After all, nobody would want an email app that occasionally sends emails to your ex and lies about your inbox. But gradually the models will get better. Even as they push further into the space of brand new experiences, they will slowly become reliable enough to use for real work.

Little pieces of this future already exist. Years ago Jonas Degrave showed that ChatGPT could do a decent [simulation of a Linux command line](https://www.engraved.blog/building-a-virtual-machine-inside/). In a similar vein, websim.ai uses an LLM to generate web sites on demand as you navigate to them. [Oasis](https://oasis-model.github.io), [GameNGen](https://gamengen.github.io) and [DIAMOND](https://diamond-wm.github.io) train action-conditioned video models on single video games, letting you play e.g. Doom inside a large model. And [Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) generates playable video games from text prompts. Generative UI may still be a crazy idea, but it's not *that* crazy.

There are huge open questions about what this will look like. Where will generative UI first be useful? How will we share and distribute the experiences that we make by collaborating with the model, if they only exist as a large model's context? Will we even want to? What new kinds of experiences will be possible? How will any of this actually work? Should models generate UI as code, or generate raw pixels directly?

I don't know these answers yet. We will simply have to experiment and find out.

> Thanks to [Michael Chang](https://x.com/mmmbchang?s=21&t=z068JnyK1Ebi5cT1b1tUEA) for discussions that inspired this whole line of thought. Thanks also to [Jonas Degrave](https://www.engraved.blog) and [Cinjon Resnick](https://x.com/cinjoncin) for feedback and discussion about earlier versions of this post.



[^notebooklm]: Special mention goes to NotebookLM's podcast feature, which takes the seemingly-trivial step from model-as-person to model-as-*two*-people. And it's amazingly different! Even a tiny change in the metaphors we use can lead to transformational shifts in experience.
[^voice-assistants]: Voice assistants have struggled with this since their creation. We have collectively figured out that "set a timer" is something Siri can do and "book me a flight to New York" is not, but this is not at all clear at first glance. Lots of things, like "change a setting", still live in the maybe-maybe-maybe gray area.
