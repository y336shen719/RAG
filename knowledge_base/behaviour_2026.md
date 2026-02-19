**[Self-introduction/ please tell me about yourself?]{.underline}**

My name is Yiming Shen, you can call me Ryan. Thanks for having me
today.

Firstly, I am goanna to talk about my education background. Now I am
pursuing a Master's in Data Science and Artificial Intelligence at the
University of Waterloo. I also completed my Bachelor's in statistics and
computing in Waterloo as well. And I got excellent standing for all
terms. So, my academic path built me a solid foundation in machine
learning, statistics and data-driven problem solving.

Beyond academics, I've completed five co-ops as data analyst across
different areas such as the government, technology and the banking,
where I worked closely with stakeholders to clarify requirements and use
data to support their decision-making.

Personally, my passion lies in extracting insights and uncovering
patterns from messy data, deliver end-to-end solutions. And technically,
I am comfortable with Python and R for analysis, and I also had hands-on
experiences from past co-op in transforming and visualizing data with
SQL, different BI tools such as Power BI, Tableau. More recently, I am
building more production-oriented skills like I'm learning Spark and AWS
in this term to expand my skills in big data and cloud.

By the way, I also have a lot of interests and hobbies in my spare time.
I am a movie lover and also love playing badmintons, swimming and
fishing.

So, my academic background, practical work experience and personal
properties, I believe they all help me become qualified with this
position.

**[Work Experience:]{.underline}**

1.  [CIBC]{.underline}

During my co-op at CIBC, the team I worked with is End Point Print and
Digital Output, which is responsible for building and running the
networks of enterprise printers across all CIBC's office locations and
branches. A software was used to monitor the network, and all
operational data were stored in an internal SQL instance. So, my job is
to rely on those data to build dashboard and provide real-time insights
into printer network performance, usage pattern and operational
metrices. Basically, the backend is SQL instance, and the frontend is
Power BI service. So, firstly, I created SQL views to filter and
aggregate the raw data. Then I built a data pipeline by deploying an
On-premises Data Gateway to connect backend and frontend. Finally, I
delivered the end-to-end solution and dashboards with weekly scheduled
refresh. During the 8-month, I collaborated with those infrastructure
engineers closely, collected feedback, refined my products and make sure
it aligned with security policies. At the end of the co-op term, I also
did the wrap-up presentation to the leadership, and I am very glad to
get an 'outstanding' performance rating in this term.

The biggest challenge I faced during the co-op in CIBC is the refresh
performance problem caused by growing data size. The printer network
data accumulate continuously, and if Power BI re-imported the entire
history data every refresh, the refresh times would keep increasing and
could eventually time out. Since other team members are all engineers
and know little about data handling, so there wasn't a ready-made
playbook in the team, and I had to figure it out on my own. At
beginning, I used the 'incremental refresh' (RangeStart, RangeEnd) in
Power BI to update only the most recent data instead of re-loading all
historical data. But I found the refresh performance was still
inconsistent. By further investigation, the key factor is a function
called query folding: if power query steps could not be translated into
SQL query and pushed down to backend, Power BI may still pull a large
amount of data and transform it locally, which interprets the
incremental refresh. To fix it, I adjusted the data transformation
steps, for example, I put filtering step as early as possible, and also,
I avoided some complex transformation. I optimized the entire workflow
to make sure most of the heavy data work were did in SQL and Power BI
only read a smaller processed dataset. As a result, the dashboards had a
fast and stable weekly refresh performance.

2.  [Kaggle Competition: S&P 500 Stock Price Predication]{.underline}

S: One project I am proud of is an ML pipeline Kaggle competition I
attended. It is about S&P 500 stock price forecasting.

T: Basically, I need create a strategy within 2 month and the strategy
will be used to predict an optimal allocation of funds to holding the
S&P500. Before this project, I knew little about quant, financial
analysis, and I indeed faced a lot of challenges, but I started from
scratch and self-learned a lot to overcome those challenges.

> \[[After submission, your strategy will run on 180 days of real market
> data. The team with the highest adjusted Sharpe ratio will be the
> winner. Sharpe Ratio: a financial metrics that indicates how much
> extra return you get for each unit of risk you take---higher is
> better]{.underline}\]

A: At the beginning of the project, I break the entire work into a few
milestones with mini deadline for each to have a better control of the
timeline. Basically, like week1 exploratory data analysis, then moved on
to week2 data preprocessing, feature engineering within 2 weeks and
finally model fitting, evaluation and so on. Now I have a very
high-level framework. Then I initialed some sub-tasks under each
milestone based on my experience, for example, under EDA, I have learned
correlation analysis, missing value analysis, target feature analysis
from schoolwork, and then I did some research online to see what else
can I do, I found that Hurst Exponent Analysis

> \[[Hurst exponent analysis checks past data to tell whether a time
> series tends to keep trending, go back to its average, or just a
> random walk]{.underline}\]
>
> is very common and useful in time series data, so I self-learned what
> is Hurst exponent analysis, why we need that. Finally, I have a very
> detailed roadmap, and I just need to follow it strictly.
>
> One of the biggest challenges is the data leakage issue in time
> series. I noticed that the validation score was unusually good, which
> raised me a red flag. It was very likely the over-confidence caused by
> data leakage. And I suspect the data leakage is coming from the new
> features introduced by feature engineering.
>
> I break the pipeline into components and conduct small, controlled
> experiments for each: changing only one component at a time while
> remaining the rest unchanged. When I removed the wavelet decomposition
> features, the score immediately fall back to a more reasonable range.
> After the online research and further investigation, I found that the
> wavelet decomposition rely on a symmetric time window, which means
> that at time index t, the window will refer to t+1, t+2's information,
> this is a very classic look-ahead in time series. To fix it, I changed
> the time window to be one-sided, strictly casual-inference to prevent
> data leakage. And then the validation score became more conservative
> but make sense.

3.  [Horizn Studio]{.underline}

Horizn is a fintech company, which built a platform to help its clients,
different financial institutions, to educate the customers and promote
digital banking. And my routine work is to collect usage data from GA4
and deliver weekly reports to our clients. There are also some ad-hoc
tasks, such as using Tableau to create dashboards, conducting
correlation analysis to figure out the reasons for sudden change in
platform usage, offering actionable suggestions to the Customer Success
Team. (Tutorial completion rate is correlated with the video length, DAU
is correlated with the content publish frequency, so we can simply the
video but increase the publish frequency, DAU is correlated with the
seasonality, tax season, so we can avoid rolling out new
features/version upgrade during this period to prevent breakdown and
improve user experience) During this co-op, I practiced how to turn
messy product-usage data into stakeholder-ready insights.

4.  [York Region]{.underline}

During the co-op at York Region, I worked as a member of the data team
and was responsible for building dashboards to visualize residents'
public transportation behavior within York Region. The backend is the
data warehouse in SQL server, and the frontend is the interactive Power
BI dashboards. A significant portion of my work involved stakeholder
communication to understand business requirements, and to collect
feedback, and keep iterating on the products until the final
deliverables met user needs. The final products were published to York
Region's internal SharePoint site and subscribed by many stakeholders,
like help the Traffic Planning team to do decision-making. This co-op
allowed me to conduct my first end-to-end ETL pipeline and practiced my
cross-functional collaboration, client-facing communication.

**[Soft skills]{.underline}**

1.  **[Explain technical concept to non-technical person]{.underline}**

-   identify my audience:

> who they are; what decision they are trying to make
>
> make my explanation to align with their business objective and domain
>
> talk to operational team: focus on implementation process; workflows
>
> talk to leaderships: business impact; risk

-   Data visualization:

> align with how people naturally think
>
> Example:
>
> Kaggle competition about S&P500 Stock Price Predication
>
> Evaluate the model's performance in real market: Sharpe Ratio\
> Our model achieved Sharpe ratio of 0.6
>
> good risk resistance
>
> simple line chart with two lines: one for the market price movement
>
> one for our strategy's cumulative return
>
> audience clearly see that during a period when the market experienced
> fluctuation, our return is still stable

-   check for understanding continuously:

> Ask 'any question so far' frequently
>
> Prevent people accumulating confusion till the end
>
> When people said they are struggling to follow
>
> Low down pace, other words to rephase, use example

2.  **[Communication and Interpersonal]{.underline}**

-   Active listening skills:

> be a quiet but active listener is beneficial
>
> Fully understand speakers' purpose and intention
>
> e.g. helped me be fully aware of stakeholders' business requirement,
> demand
>
> Give me a moment to think carefully before I response
>
> Avoid misunderstanding

-   Identify the audience

-   Clear communication

> Try to make communication as clear and content-driven as I can

***S***: last coop in CIBC, I worked as a member of the team with around
20 IT engineers

***T***: My role required frequent communication with both internal IT
engineers and non-technical leaderships.

***A***: pay attention to the feedback loop: collect feedback after I
deliver some work, iterate/refine based on feedback. This active
listening helped me capture their demand and avoid unnecessary rework.

Before every week's standup meeting/ some prensentation, I would draft a
short outline, maybe just a few bullet points to write down the key
things I want update. Also, I would initial 1-2 questions people might
ask and prepare their short answers.

***R:*** These measures all keep my communication efficient and to the
point.

3.  **[Time management]{.underline}**

-   Always start early:

> Unpredictable, unexpected change come up
>
> Requirement change, data quality issues
>
> Uncover issues in advance instead of waiting till the last minute

-   Scheduling & planning

> Break a task into multiple milestones
>
> Set up mini-deadlines for each and list out timelines
>
> Using planning tool, apps to keep everything on track e.g. outlook
> calendar, simple to-do-list

***S***: During my coop at Horizn as a data analyst

***T***: assigned with 3 ongoing tasks, timeline was tight

**A**: Reached out my leader, figure out the priorities and deadlines

Some tasks were client-facing with hard deadlines

Did time distribution based on priorities and deadlines

Usually, client-facing task with earlier hard deadline, do first

With higher priority, invest more time on it to ensure the quality

Lower priority but time consuming, time-boxed the work and provide
progress update

*R:* was able to keep all 3 tasks moving forward and deliver them on
time with high quality

4.  **[Multitasking skills]{.underline}**

-   Efficient: try to work smart, not just work hard

> get more things done in less time

Skip unnecessary steps to simplify workflow

-   Have time conscious and use time management to balance time properly

> (Same as above time management)

5.  **[Stress handling ability]{.underline}**

-   I found myself is actually quite resistant to stress

Stressful, anxious when I faced multiple challenges at the same time and
deadlines are approaching

Before midterm: co-op interview, assignment deadlines and exam schedules
all crashed together

-   Use scheduling & planning

> Stress caused by uncertainty
>
> Break big task small piece, milestones
>
> Feel have a better control
>
> Follow the roadmap step by step
>
> focus on work
>
> less stressful

-   Personal strategy to handle stress: tidy up my room, especially work
    desk

> Clean, tidy work environment reduce anxiety
>
> Make me feel comfortable and easily focus on work instead of worrying

-   Use hobbies to relieve stress: play badminton, swimming, go hiking
    with friends, watch a movie

    1.  **how to handle the stress when the team has a deadline
        approaching (team perspective)**

-   recommend quick, short update

> no need be detailed, objective-oriented
>
> just a brief text on teams is fine
>
> point is making every team member feel engaged
>
> in a team, when deadlines approaching
>
> for teamwork, it is common that my work is relying on someone else's
> output, I might need wait for someone or someone need waiting for you
>
> feel like not a member but working individually
>
> Without communication/updates, the waiting will increase anxiety
> within the team
>
> Increase transparency, clarity within team
>
> deadline becomes more manageable

1.  **How to handle stress when there is a deadline approaching
    (individual perspective)**

> Feel stress because of the uncertainty
>
> How much work left, am I able to catch the deadlines

-   reduce uncertainty by a clear plan/schedule

> break the work into small pieces, like many achievable milestones
>
> Set mini deadlines for each

-   If I see a risk of missing the deadline

> Will not hesitate to ask for assistance from other team members
>
> skip some unnecessary steps to simply workflow
>
> Delivery MVP (minimum viable product) first
>
> Take rest of the time to polish/refine it
>
> Uncover issues in advance instead of waiting till the last minute

6.  **[Failure handling ability]{.underline}**

-   Upset and frustrated

```{=html}
<!-- -->
```
-   Take short rest

```{=html}
<!-- -->
```
-   Control you emotion, Do not make negative emotion caused by failure
    effect you for a long time

```{=html}
<!-- -->
```
-   Convince myself, told myself in mind: nobody can always succeed

```{=html}
<!-- -->
```
-   Reflection and summary

-   Use failure as a path to improve

-   Make some notes, documents to avoid making similar mistakes again

7.  **[Conflict handling ability]{.underline}**

-   Conflict occurs when there is a disagreement

-   Figure out the root cause of disagreement

> Ask 'why' like 'can you walk me through your reasoning?'
>
> provide the other person with some space to explain their thinking

-   sometimes not just simply one is right, the other is wrong

> People reach different conclusions because they sit in different
> positions
>
> try to put myself in their shoes to think from their perspective

-   Keep arguing is meaningless

> Provide evidence to back up / provide options to move on
>
> Evidence: use data, real case
>
> Options: conduct a small A/B test, small experiment to make decision

***S***: when I worked with other teammates on the Kaggle competition
S&P500 Stock Price Predication

***T***: A conflict/disagreement took place at the stage of feature
engineering

I argued: create and add new features

Some teammates: reduce dimension by using like PCA

***A***: Set up a 30-min discussion

Let both sides illustrate their reasoning

> My opinion was: create new, informative features enhance the signal
> that model can learn; if worry about overfitting, we can add strong
> regularization, like L1 LASSO can shrink the parameters of
> insignificant features to 0
>
> The other side: stock price has very low signal-to-noise ratio, adding
> new features very likely overfitting noise; reduce dimension can avoid
> overfitting, reduce complexity
>
> Both sides are reasonable
>
> No further discussion, we designed a small, controlled experiment
>
> Keep everything else same: same base model, same walk-forward
> validation method

***R***: we found the overall performance are close but adding new
features one is more stable in each fold

Reach agreement to add new features and move on

1.  Disagree with your colleagues/supervisors

-   Communication:

> Transparency, visibility within the team is important
>
> Clarify questions; make sure both are on the same page; ask reasons
>
> In some cases, it could be there is something I missed, especially
> have different opinion with supervisor/more experienced colleagues

-   Stop arguing, either using evidence to back up, or providing options
    to move on

> If there is an issue:
>
> explain and persuade in a polite and respectful way
>
> back up evidence, like data
>
> If disagreement is about approach:
>
> Propose quick & low-cost experiment (small subset of data) to make
> decision
>
> Instead of arguing

8.  **[Teamwork]{.underline}**

-   Visibility and transparency of information

```{=html}
<!-- -->
```
-   Communication

-   Feedback loop

```{=html}
<!-- -->
```
-   Diversity

> Diversity encourgaes innovation
>
> welcome and open to different voices

-   Documentation

> Document key decision, discussion result, action

-   Prevent miscommunication

> Keep everyone on the same page

-   Reduce rework

***S***: During my last co-op at CIBC,

***T***: work as a member of team with around 20 IT engineers

***A***: communication

Visibility & transparency

> would like to share any updates, even minor progress in team stand-up
> meeting
>
> raise the challenges I faced and ask for guidance / assistance when
> needed
>
> as a part of the team, be ready to provide others with assistance
>
> feedback loop

***R*:** I developed strong trust and smooth collaboration with other
team members

9.  **[Problem solving]{.underline}**

-   Know how to make the full use of any available resources

```{=html}
<!-- -->
```
-   Documents, well-made tutorial, YouTube, tech community

-   Ask colleagues for support when needed

> e.g.

***S***: At York Region, I was new to Power BI

***T***: want to ramp up quickly

***A***: I found a workshop arranged by Microsoft: Dashboard in a Day

After supervisor's permission: enrolled in this training

***R***: cleared up a lot of my confusion, like modeling, DAX script.
Became very beneficial in future work

-   Diagnostic mindset + learning mindset

```{=html}
<!-- -->
```
-   Diagnostic: break the problem into small pieces/components, run
    small experiments on each isolate and locate root cause

-   Learning: bottleneck is a knowledge gap, proactively learn to close
    that gap and apply what I learned into solutions

***S***: At CIBC, build a BI reporting pipeline, BI service will load
data from SQL instance to run scheduled refresh

***T***: growing data size caused refresh performance problem

As data accumulate continuously, full refresh became slower and
eventually time out

***A***: after research online, I found incremental refresh could be
potential solution

Self-learned how to implement the incremental refresh via online
tutorials

After deployed, I noticed the refresh performance was still not improved

By diagnostic mindset, I break into 3 components to isolate and source
the root cause: power BI layer, gateway layer and SQL layer

By further investigation, the key issue is a concept called query
folding in BI layer

failure of query folding interpreted the incremental refresh

***R***: I fixed the issue and increment refresh was working as intended

10. **[Goal reaching ability]{.underline}**

-   Scheduling & planning

> Set a long-term goal
>
> Break the work into many small achievable milestones with priorities
> and mini-deadlines
>
> A timeline to reach the goal

-   Persistence and strong executive power

> follow the plan as scheduled
>
> even I encounter some blockers
>
> adjust approach and generate alternative solutions

11. **[Detailed-oriented / Attention to accuracy]{.underline}**

-   Make data roadmap before start work

> Write down any details are easy to be missing or supposed to notice

-   Do (accuracy) validation and sanity check before moving on

> Prevent bringing accuracy issue to next steps
>
> Cross check: use different methods to verify your output independently
>
> e.g. in data reporting pipeline, for a KPI displayed on Power BI
> dashboard, run a SQL view to verify

12. **[Self-motivation / self-starter]{.underline}**

-   Jumped out of comfort zone to enrich my skillsets

-   Work independently with little supervision

***S***: co-op at YR, we have sprint showcase regularly, people can
share their work, their idea to broader audience in the company

***T***: public speaking is not my comfort zone, used to avoid
presentation in front of a large group of people

***A***: To help myself step out of this comfort zone, I attended many
sprints showcase early on as a listener, to learn how other people
structure their demo

At the end of the co-op term, I delivered my final presentation at the
sprint showcase, introducing my product to the entire company.

***R***： It help my product gain much more subscription from other
teams

13. **[Adaptability]{.underline}**

-   Fast learner

***S***: co-op at Phillips Lighting, I worked as a procurement intern,
the work content was totally different from my academic background.

***T***: I don't have any knowledges about supply chain; about those
electronic components they purchased.

***A***: In order to adapt to this position quickly, I did a lot of
self-learning. Every time I ran into a problem about some business
definition, I would reach out my colleagues and figure it out. Also, I
would take some notes to write down the answers.

***R***: after one month, I was able handle every task assigned to me. I
get used to this new position much faster than my manager's expectation

***Additional***: looking at my work history, we will find that I have
worked across different teams and domain, but every time I am able to
adapt to new environment quickly and can pick up industry knowledges
fast enough

-   Make full use of any available resources, even sometime the resource
    not presented in front of me, I need search for it proactively

***S***: At York Region, I was new to Power BI

***T***: want to ramp up quickly

***A***: I found a workshop arranged by Microsoft: Dashboard in a Day

> After supervisor's permission: enrolled in this training

***R***: cleared up a lot of my confusion, like modeling, DAX script

1.  **[How to ensure the accuracy of data?]{.underline}**

-   Identity the possible sources of inaccuracy:

> Such as when switching units: from kg to g
>
> Some invertible sources: bias, deviation
>
> N/A (unknown value)
>
> Be careful at those places

-   Technical strategies for checking accuracy:

> Confidence interval, likelihood interval

-   In many cases: data are correlated, check if everything match

> e.g. calculate number of household in each municipalities (Markham,
> Georgina) within YR
>
> add them up altogether, if summation match the YR's total household
> number

2.  **[How to conduct a report?]{.underline}**

-   Identity the target reader/audience:

> Know what they wish to get from this report
>
> Leader, colleagues: what you have done, what progress you have made,
> how to do future maintain
>
> Client, stakeholder: your product's features, how to use

-   Clear structure and high-level outline:

> PPDAC: problem, plan, data, analysis and conclusion
>
> Easy for reader to catch the key points

-   Prefer to use diagrams, numbers to support my points, conclusions

> Easy to understand and convincing

3.  **[How to connect with clients?]{.underline}**

-   Concise, clear and context-driven

-   Active listener: Fully aware of their requirement and demand

-   Reaching out frequently: update any progress you've done

> Let them know everything is on track

-   Involve them into your project

> Will be happy if they got a sense of involvement
>
> Ask for their feedback, how's their feeling
>
> When cannot sure what they really like: provide option A, B, let them
> make choice

4.  **[What is your experience of visualizing data?]{.underline}**

-   Workplace: use Power BI

> Load processed data into Power BI for visualization
>
> Build interactive dashboards
>
> Different visuals to illustrate data, slicers control each visual
>
> Create hierarchy to enable user to get deeper into our data
>
> All products published on platform, accessible to everyone in YR

-   Courses: use R

> Boxplot and relative frequency histogram: distribution of data
>
> QQplot, scatterplot with fitted line: check how well the model fits
> data

5.  **[What is your experience of using Excel?]{.underline}**

-   Process worksheets with Excel and provide the team with data support

-   Vlook-up, pivot table, functions, filtering and sorting

6.  **[How to process vague data]{.underline}**

-   Understand data

-   Clarify: what kind of decision that our stakeholders try to make

**[Questions:]{.underline}**

1.  This is a warm up question so you get the hang of the video
    interview process. What is something you have achieved within the
    past year that you are really proud of? You have one minute

Within the past year, I independently completed a research on analyzing
the twitter activities for health agencies during Covid-19 pandemic and
helped them formulate social media strategy. It took me over 2 weeks to
complete the project, from sampling data, building statistical model to
conducting a twenty-five thousand words report. I achieved A level on
this project, which makes me really proud.

2.  Why did you choose your particular academic program?

Currently, I am studying in Statistic, Co-op program at the university
of waterloo. The reason why I chose this academic program is that I plan
to become a data scientist and do jobs related to big data after
graduation, and I am making efforts to let it comes true. Due to
computerization, almost all industries need data supporting, the
financial industry, manufacturing industry and energy industry as well.
So I believe that big data is a trend with bright future. This program
can teach me how to analyze and extract useful information from a large
amount of messy data. It is very meaningful. That is the reason why I
chose this program.

3.  Please tell us about the work experience, volunteering or
    extracurricular activities that may have prepared you to work at
    Enbridge. What were you doing and how do you feel it has prepared
    you to excel in a co-op at Enbridge?

During last co-op term, I worked as a procurement intern in Philips
Lighting and was responsible for data support and making price lists
with Excel. So I have a high proficiency in operating Excel, Words and
any other office software. Since managing prices took up most of my
daily work, so it helps me form my strong attention to details and
accuracy, which will be very helpful and essential for this job,
especially since this job includes monitoring time and schedule. Besides
that, my strong communication skills helped me complete over 10 RFQ,
which is request for quotation project from various suppliers via email
and phone call, so such communication skill will also be a great assert
to this position.

**[Behavior Questions Bank]{.underline}**

1.  [Why Are You the Best Person for the Job/ Why you want this
    job?]{.underline}

> the work aligns with what I enjoy most: taking messy data, extracting
> meaningful insights, and turning them into actionable solutions.
>
> Use data to bring chaos into order
>
> strong fit for my background and a great opportunity to apply what
> I've built so far
>
> So, I feel confident that I can contribute quickly and deliver impact
>
> I'm excited about the growth path here
>
> based on the job posting/introduction, I'm especially eager to deepen
> my skills in
>
> a natural next step in my transition from data analyst to data
> scientist

2.  [What interested you about this position?]{.underline}

3.  [What do you know about this company?]{.underline}

4.  [What did you enjoy most about your last position?]{.underline}

5.  [What did you learn from last position?]{.underline}

> Problem-solving

6.  [How do you handle success/ failure?]{.underline}

> Do not let negative emotion affect you for too long
>
> Feel better after realized the facts
>
> Do reflection and summary
>
> I also always try my best to avoid making a same mistakes for twice

7.  [How well do you work with others?]{.underline}

> Communication
>
> Feedback loop
>
> Ask for guidance from
>
> Ready to provide support

8.  [What is your greatest weakness?]{.underline}

Leadership

Improving now

9.  [What is your greatest strength?]{.underline}

Adaptability

Fast learner, be quick to make data science rooted in industry

Know how to make the full use of any available resource to me

10. [What is your leaders' feedback to you?]{.underline}

In the final performance judgement: outstanding

Self-motivated

Fast learner, handle the work much faster then she expected

11. [What are you future goals?]{.underline}

> Short-term: finish my master degree, take more courses about
> supervised learning, deepen my knowledges in computer vision, this is
> a trend. Also, I will develop much more production-oriented skills:
> deploy and run the model in real production environment
>
> Long-term: become data scientist, this is my passion so far

12. [What is your future plan?/ what do you see yourself in future 5
    years?]{.underline}

> first 1 years: finish my master degree
>
> next 4 years: I want to spend three-year building very solid
> foundation as a data scientist and then transited into an MLE, so I
> will be very familiar with the entire model lifecycle.
>
> More importantly, no matter where I am, what my title is, I want to do
> something I really interested about, where my passion lies. Because
> passion, curiosity drivers me to keep learning, keep myself up-to-date
> in that area.

13. [What are the essential skills of a proficient worker?]{.underline}

> Strong business sense
>
> Explain technical to non-tech

14. [Describe a difficult work situation on how you overcame
    it.]{.underline}

> multitasking

15. [What is your biggest professional achievement?]{.underline}

16. [Do you prefer to work independently or on a team?]{.underline}

> both is okay
>
> different benefits
>
> avoid being affected by other people, can focus on your own work
> better
>
> prefer teamwork at beginning of project
>
> then work individually

**[Situation Questions Bank]{.underline}**

About cost-saving: substitution

1.  [Tell me about a long-running project you handled. How did you
    manage your time to meet your deadlines?]{.underline}

2.  [Describe a situation where you saw a problem and took steps to fix
    it.]{.underline}

3.  [Tell me about a tough challenge you faced. How did you solve
    it?]{.underline}

4.  [Tell me a time you are stressful.]{.underline}

5.  [Tell me a time that you failed.]{.underline}

> [Many time I thought myself work really hard, but the result is not
> match my expectation]{.underline}
>
> [A STAT course, I spent three days to review and I thought I invested
> a lot of efforts and was fully prepared]{.underline}
>
> [If the truth is you have already done everything you could, then
> maybe you just need persist on it]{.underline}
>
> [Because the learning curve is not a straight upward line, you will
> face bottleneck]{.underline}

6.  [Tell me a time that you reached a goal.]{.underline}

7.  [Tell me a time that you were upset.]{.underline}

8.  [Tell me a time that you collaborated with others]{.underline}

9.  [Tell me a time that you had conflict with your
    colleagues.]{.underline}

10. [What will you do if you have a disagreement with your
    leader?]{.underline}

11. [Tell me a time that you need learn new thing within a short
    deadlines.]{.underline}

> Latex
>
> Not common, never heard before
>
> It is about just one day before the deadline
>
> Professor suddenly emailed everyone
>
> bonus mark if you can writing your answer with latex
>
> A new programming language, transfer some code into mathematics
> notation
>
> In order to learn fast
>
> I learn and use them at the same time.
>
> Positive feedback loop

12. [Tell me about a time you had to collaborate with a coworker who was
    difficult to work with.]{.underline}

> I keep good relationship with everyone I have worked with
>
> But this is one colleague that is a little bit difficult to work with
>
> He preferred working individually instead of the teamwork
>
> That lead to the lack to communication
>
> Feedback loop
>
> Documented properly all the decision we made together, action we take
>
> Reduce miscommunication
>
> Make sure we are always on the same page
>
> Reduce a lot of unnecessary rework

13. [Tell me about a time you went above and beyond for
    work.]{.underline}

14. [Tell me about a situation when your job went through big changes.
    How did you adjust?]{.underline}

> [Assigned with 3 ad-hoc tasks on short notice]{.underline}
>
> [Learn within a short time]{.underline}

15. [How to obtain information from others?]{.underline}

```{=html}
<!-- -->
```
17. [Please tell us about the work experience, volunteering or
    extracurricular activities that may have prepared you to
    this.]{.underline}

18. [Negative feedback]{.underline}

19. [Tell me a time you get a feedback]{.underline}

> Outstanding in final evaluation
>
> Highest rating in Uwaterloo's work term evaluation
>
> Some writing materials are needed to support
>
> Self-motivated
>
> Able to work with little supervision
>
> I go beyond the scope of my job
>
> I discovered how to embed the dashboard into SharePoint site
>
> I used Power Automate to automate the log reporting and trigger
> warning when issues occurred
>
> No one even asked me to do so

20. [Make an important decision]{.underline}

21. [What do you do to stay motivated with sometimes mundane
    tasks?]{.underline}

> [Passion, data work is boring, data cleaning]{.underline}
>
> [Go beyond the scope of task, innovation]{.underline}
>
> [Automate the repeated work]{.underline}
>
> [Apply front edge technology]{.underline}

22. [A project you found technically or conceptually challenging, and
    what you learned from it]{.underline}

23. [Experiences receiving feedback or navigating ambiguity]{.underline}

> [Stakeholders' business requirement is unclear and vague]{.underline}
>
> [Reach out immediately, clarify it, because I don't want to bring
> unclarity into next step]{.underline}
>
> [Create something completely, and even the customers clearly define
> what they want]{.underline}
>
> [Deliver MVP first: uncover issues in advance, more easier to provide
> feedback based on the MVP]{.underline}

24. [How you approach learning new skills and taking
    ownership]{.underline}

> [Use it first then master it]{.underline}
>
> [For a lot of theoretical knowledge, only when you apply it, you will
> learn really fast]{.underline}
>
> [Persistence: learning curve is not a straight line upwards, you will
> meet bottleneck in the middle way, you need persist on it instead od
> giving up]{.underline}
>
> [Make full use of available resources]{.underline}
>
> [Taking ownership]{.underline}
>
> [Treat work as an end-to-end outcome, not just a task]{.underline}
>
> [During the work, feedback loop]{.underline}
>
> [Because I take the ownership of the product, instead of waiting
> someone else reach out me]{.underline}
>
> [After I delivered the work, I did very detailed documentation,
> because some else need maintain it]{.underline}

25. [How you use AI]{.underline}

26. [What improvement you can do in one of the coop/project you
    did]{.underline}

> [During my coop at horizn]{.underline}
>
> [Use more advanced ML model to replace traditional stat
> analysis]{.underline}
>
> [Correlation analysis: use existed data to explain something already
> happened]{.underline}
>
> [ML: predict future]{.underline}

27. Tell me about a time your work was criticized

> I present my first version of work, my manager not satisfied, it can
> be better
>
> [Do reflection]{.underline}
>
> [I collected feedback from each one individually]{.underline}
>
> [Stakeholder is not a single person, a team]{.underline}
>
> [There lacked the communication between stakeholders]{.underline}
>
> [They did not reach an agreement/consistency on their business
> requirement]{.underline}
>
> [Requirements are conflicted]{.underline}
>
> [Create a shared online doc with version control, invite everyone
> write down their requirements]{.underline}
>
> [People can see each other's requirements]{.underline}
>
> [Conflict: open doc on stand up, set up a 5-10 mins quick discussion,
> until reach agreement, move on]{.underline}
>
> [Make each's business requirements more visible and
> traceable]{.underline}

<https://igotanoffer.com/blogs/tech/amazon-behavioral-interview#questions>

<https://igotanoffer.com/blogs/tech/behavioral-interview-questions#company>

<https://resumegenius.com/blog/interview/behavioral-interview-questions?utm_source=google&utm_medium=performancemax&utm_term=&utm_campaign=18128645407&utm_adgroup_id=&utm_content=&utm_device=c&gad_source=1&gad_campaignid=18133332233&gbraid=0AAAAAD-TJkCviZ5YJizxFFcohGCKBm0es&gclid=Cj0KCQiAyvHLBhDlARIsAHxl6xrOqnYWfBiSbpFv9b15uLea-JBo1CkZFSsOJ3BUC0bmqtPZr1OZqIkaApWAEALw_wcB>
