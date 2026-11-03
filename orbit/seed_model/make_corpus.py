"""Create an original, synthetic multi-turn training corpus for ORBIT Seed.
All text is authored/generated locally for this project; no pretrained model or copied corpus.
"""
from pathlib import Path
import random

R = random.Random(73421)
OUT = Path(__file__).with_name('corpus.txt')

names = ['Mira','Jonah','Nia','Eli','Sasha','Rowan','Tess','Amir','June','Theo','Iris','Caleb','Rin','Morgan','Avery','Noah']
skills = ['drawing','programming','gardening','cooking','writing short stories','playing guitar','photography','learning Spanish','making small games','repairing bicycles','building model kits','playing chess']
projects = ['a small platform game','a neighbourhood garden','a personal website','a simple weather station','a short mystery story','a study timetable','a wooden bookshelf','a beginner robot','a community newsletter','a tiny puzzle game','a photo portfolio','a model railway']
interests = ['space exploration','old architecture','wildlife','music theory','ocean science','history','game design','botany','film making','electronics','maps','mythology']
obstacles = ['I keep losing track of the next step','I start enthusiastically and then get stuck','I have very little free time','the instructions assume I already know the basics','I keep changing the design','I am not sure how to test my progress','I get overwhelmed by too many choices','I made a mistake and do not know where it happened']
minutes = [10,15,20,25,30,40,45,60]
places = ['a quiet library','a busy train station','a greenhouse','a small coastal town','an old workshop','a mountain cabin','a museum after closing','a floating research station','a market square','a lighthouse']
objects = ['a brass key','a cracked compass','a paper crane','a blue notebook','a tiny radio','a glass marble','a wooden bird','a faded photograph','a silver button','a map with a missing corner']
animals = ['fox','raven','otter','cat','turtle','hare','heron','dog','badger','squirrel']
colors = ['amber','indigo','copper','green','silver','violet','coral','blue']

lines=[]
def add(turns):
    lines.append('\n'.join(f'<|{role}|> {text}' for role,text in turns) + '\n<|end|>')

# Conversational coaching with a persistent project/goal across multiple turns.
for i in range(900):
    n=R.choice(names); skill=R.choice(skills); project=R.choice(projects); mins=R.choice(minutes); obstacle=R.choice(obstacles)
    style=R.randrange(4)
    if style==0:
      turns=[('user',f"I want to get better at {skill}, but {obstacle}."),('assistant',f"Let's make it manageable. Choose one small outcome for today rather than trying to master {skill} all at once. What would you like to be able to do first?"),('user',f"I'd like to use {skill} to work on {project}."),('assistant',f"That gives us a useful target. For {project}, the first milestone should be a rough, working version. You can improve it after you have something concrete."),('user',f"I can spend about {mins} minutes a day. Is that enough?"),('assistant',f"Yes. A focused {mins}-minute session is enough to make steady progress. Spend a few minutes reviewing, most of the time practising, and the final minute writing down the next step."),('user',"What should I do tomorrow?"),('assistant',f"Continue with {project}. Repeat the first exercise, then add one small feature. Keep the task short enough that you can finish it in one session.")]
    elif style==1:
      turns=[('user',f"My name is {n}. I'm learning {skill}."),('assistant',f"Nice to meet you, {n}. What made you interested in {skill}?"),('user',f"I want to create {project}, but I am still a beginner."),('assistant',f"Being a beginner is fine. Break {project} into pieces and learn only the skill needed for the next piece. What part feels most confusing?"),('user',obstacle.capitalize()+'.'),('assistant',f"Let's isolate that problem. Describe what you expected, what happened instead, and one thing you have already tried. We can work through it without changing everything at once."),('user',"Can you remind me of the plan?"),('assistant',f"Your goal is {project}. Work on one small part of it during each practice session, record what changed, and decide on the next action before you stop.")]
    elif style==2:
      turns=[('user',f"I have an idea for {project}."),('assistant',"Tell me the rough version. It does not need to be polished."),('user',f"I am interested in {R.choice(interests)}, and I want the project to be simple enough for a beginner."),('assistant',"A good first version should demonstrate the central idea and leave optional features for later. What is the one thing a person should be able to do with it?"),('user',f"I think they should be able to explore it for {mins} minutes and discover something."),('assistant',"Then make exploration the main loop. Give the user a clear starting point, a few meaningful choices, and feedback when a choice changes the situation."),('user',"What if the first version feels too small?"),('assistant',"A small finished version is more useful than a large unfinished one. Test the core experience first; add complexity only when it improves that experience.")]
    else:
      turns=[('user',f"I keep thinking about {R.choice(interests)}. Could I turn that into {project}?"),('assistant',"Yes. Start by choosing a narrow question about the subject. A project becomes easier to shape when it has a clear question to answer."),('user',f"I have {mins} minutes each day and I don't want the plan to become stressful."),('assistant',"Use a flexible routine. Set a small minimum task, and treat extra time as a bonus. Missing one day should not reset your progress."),('user',"How do I know when it is good enough?"),('assistant',"Write down two or three simple success criteria before you begin. When the project meets them reliably, call the first version complete. You can always make a second version later.")]
    add(turns)

# Everyday conversation: clarification, emotional support, planning and follow-through.
scenarios=[
 ('late bus','I am running late and the bus has not arrived.','Check whether there is a live service update, then let anyone waiting for you know that you may be delayed. If another route is available, compare its total travel time before switching.'),
 ('difficult email','I need to write an email about a problem, but I do not want to sound rude.','State the issue plainly, describe its effect, and make a specific request. Keep blame and guesses out of it. A calm, direct message is usually clearer than a long apology.'),
 ('busy week','My week is full and I cannot fit everything in.','List the tasks, identify which ones have real deadlines, and choose the three most important outcomes. Reschedule or simplify the rest instead of treating every task as equally urgent.'),
 ('new hobby','I want to try a new hobby, but I do not know which one.','Choose something inexpensive that you can try in a short session. The first goal is not to pick a lifelong hobby; it is to discover whether the activity is enjoyable.'),
 ('remembering details','I keep forgetting small details while working.','Keep a short external note with the current goal, the last completed step, and the next action. Update it when you stop. A simple reliable record is better than trying to remember everything.'),
 ('creative block','I have an idea but I cannot get started.','Make a deliberately rough first draft. Give yourself permission to leave gaps and awkward parts. Once something exists, you can improve it one decision at a time.'),
 ('learning from mistakes','I made a mistake and now I feel like I have wasted the whole afternoon.','The time was not wasted if you learned what caused the problem. Write down the failed assumption, make the smallest correction, and test again before changing anything else.'),
 ('too many options','There are so many choices that I cannot decide.','Choose two or three criteria that matter most, compare only the strongest options, and make a reversible decision. You do not need perfect information for every low-risk choice.'),
 ('teamwork','Someone on my team and I disagree about the design.','Ask what problem each design is trying to solve. Compare them against the same requirements, and build a small test if the difference is difficult to judge from discussion alone.'),
 ('rest','I feel guilty when I take a break.','Rest is part of maintaining attention. Take a real break, then return with one specific next task. Constant effort is not the same as useful progress.'),
]
for i in range(650):
    topic, user_msg, answer=R.choice(scenarios)
    follow=R.choice(['That makes sense. What is the first step?','Could you make that more specific?','What should I avoid doing?','How can I tell whether it worked?','Can you give me a short example?'])
    if follow.startswith('That makes'):
      second='Start with the smallest action that changes the situation. Make it concrete enough that you can complete it without solving the entire problem.'
    elif follow.startswith('Could'):
      second='Write down the current situation in one sentence, choose the next action, and set a clear stopping point. Then review the result.'
    elif follow.startswith('What should'):
      second='Avoid making several changes at once or assuming you know the cause before checking. Small tests make it easier to understand the result.'
    elif follow.startswith('How can'):
      second='Decide what success looks like before you start. Afterward, compare the result with that measure rather than relying only on how it feels.'
    else:
      second='For example, replace “make progress” with “write the opening paragraph” or “test one feature”. A specific action is easier to begin and evaluate.'
    add([('user',user_msg),('assistant',answer),('user',follow),('assistant',second)])

# Explanations written for a curious beginner.
explanations=[
('What is a computer program?','A computer program is a set of instructions that tells a computer how to perform a task. The instructions are written in a programming language, translated into a form the computer can execute, and carried out by the processor. A program can read input, make decisions, store information, and produce output.'),
('What is an algorithm?','An algorithm is a finite sequence of steps for solving a problem. A recipe is a familiar example: it describes inputs, an order of operations, and an expected result. A useful algorithm is clear enough to follow and handles the cases it is supposed to support.'),
('What is memory in a computer?','Computer memory stores information that a system needs. RAM is fast working memory used while programs run, and its contents are usually lost when power is removed. Storage such as an SSD keeps files when the computer is turned off. They serve different purposes.'),
('What is a neural network?','A neural network is a mathematical system made of connected layers of adjustable values called parameters. During training, it receives examples, makes predictions, measures its errors, and adjusts its parameters. Repeating this process can teach it statistical patterns in the examples.'),
('What does training a language model mean?','A language model is often trained to predict the next token in a sequence. It sees part of a text, predicts what comes next, compares that prediction with the actual next token, and updates its parameters. Repeating this over many examples teaches patterns of language.'),
('What is overfitting?','Overfitting happens when a model learns its training examples too specifically and performs poorly on new examples. Training loss can keep falling while validation performance stops improving. A separate validation set helps reveal when this is happening.'),
('Why do stories need structure?','A story feels coherent when events connect through causes, goals, and consequences. A character wants something, faces obstacles, makes choices, and experiences results. Details introduced early should remain consistent unless the story explains why they changed.'),
('What is a variable in programming?','A variable is a named place to store a value that a program can use. For example, a variable named score might contain the number 12. A program can read the value, update it, and use it when making decisions.'),
('What is an API?','An API is a defined way for one piece of software to communicate with another. It describes the available operations, the information they accept, and the form of the response. A clear API allows programs to work together without needing to know each other’s internal implementation.'),
('What makes an explanation useful?','A useful explanation begins with the central idea, defines unfamiliar terms, and gives an example. It should separate facts from assumptions and avoid adding detail before the reader needs it. A good explanation helps someone use the idea, not merely repeat its definition.'),
('How should I debug a program?','First reproduce the problem consistently. Read the error message, inspect the relevant inputs and state, and form one specific hypothesis. Change one thing, run the same test again, and record whether the result supports your hypothesis.'),
('What is a feedback loop?','A feedback loop uses the result of an action to guide the next action. A thermostat measures the temperature, compares it with a target, and adjusts heating. In learning, feedback helps a person or model notice the difference between the intended result and the actual result.'),
('What is probability?','Probability describes how likely an event is, usually with a value from zero to one. Zero represents an impossible event and one represents a certain event. A probability of 0.5 means an event is equally likely to happen or not happen under the stated model.'),
('What is the difference between a fact and an opinion?','A fact is a claim that can be checked against evidence. An opinion expresses a judgement, preference, or interpretation. Some statements contain both: the factual part can be verified, while the evaluation depends on values or perspective.'),
]
for q,a in explanations:
    for _ in range(24):
      add([('user',q),('assistant',a)])

# Short stories with repeated entities and clear causal continuity.
for i in range(700):
    hero=R.choice(names); place=R.choice(places); obj=R.choice(objects); animal=R.choice(animals); colour=R.choice(colors)
    goal=R.choice(['find the missing route home','repair the old clock','deliver a letter before sunset','discover who had left the lights on','return a lost object to its owner','help the stranded visitor','save the garden before the frost'])
    problem=R.choice(['a heavy rain had washed away the path','the map had one important corner missing','the battery was almost empty','the door was locked from the other side','the last train had already departed','the wind kept scattering the notes'])
    solution=R.choice([f'{hero} checked the {obj} carefully and noticed a small mark that pointed toward a safer route.',f'The {animal} appeared again and led {hero} to a sheltered passage.',f'{hero} asked a nearby worker for help instead of guessing. Together they found a practical solution.',f'{hero} divided the task into two parts and solved the easier one first. The remaining problem was no longer difficult.'])
    story=(f"{hero} was staying near {place} when a quiet problem appeared. The goal was to {goal}. At first everything seemed straightforward, but {problem}. "
           f"{hero} stopped and looked at the situation instead of rushing. Nearby was {obj}, marked with a thin {colour} line. It seemed unimportant, but it gave a useful clue. "
           f"{solution} The problem did not disappear all at once. {hero} tested the idea, checked the result, and made one adjustment. By the time evening arrived, the goal had been reached. "
           f"Later, {hero} wrote a note about what had worked: pay attention to details, ask for help when needed, and test a solution before trusting it. The next challenge would be easier because of that lesson.")
    add([('user',f"Write a short, coherent story about {hero} at {place}. Include {obj}, a {animal}, and a problem that gets solved."),('assistant',story)])

# Roleplay-style exchanges to teach stable character voice and continuity.
for i in range(500):
    char=R.choice(names); place=R.choice(places); obj=R.choice(objects); goal=R.choice(['find a safe route','repair the signal device','understand the strange message','return the object','locate the missing supplies'])
    add([('user',f"Let's roleplay. You are {char}, a careful explorer at {place}. Stay in character and remember that your goal is to {goal}. I hand you {obj}."),('assistant',f"{char} turns the {obj} over in both hands, studying its details. 'This might be useful,' I say. 'I should understand what it does before we rely on it. What have you noticed about it?'"),('user',"I tell you it was hidden under the floorboards, and there is a symbol on its side."),('assistant',f"I hold the {obj} closer to the light and look for the symbol. 'Then it was probably placed there deliberately. I will remember where we found it. First, let's compare the symbol with anything else in this room.'"),('user',"A few minutes later, we hear a noise behind the wall. What do you do?"),('assistant',f"I step away from the wall and listen without touching it. 'We should not assume the noise is harmless. I will check for a second exit while you stay where you can see the room. If we find a safe way to investigate, we can return to the symbol.'")])

# A few explicit honesty and uncertainty examples.
for _ in range(100):
  add([('user','Do you know the exact answer to every question?'),('assistant','No. I can make mistakes, and I do not know everything. I should distinguish what I know from what I am inferring. When a detail matters and I am uncertain, I should say so rather than invent an answer.'),('user','What should you do if you cannot verify a claim?'),('assistant','I should explain the uncertainty, identify what evidence would help, and avoid presenting a guess as a fact. A useful answer can still describe the best-supported possibilities.' )])

R.shuffle(lines)
OUT.write_text('\n\n'.join(lines), encoding='utf-8')
print(f'Wrote {len(lines):,} examples, {OUT.stat().st_size:,} bytes to {OUT}')
