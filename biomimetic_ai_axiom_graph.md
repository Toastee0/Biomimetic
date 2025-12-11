# Biomimetic AI: Complete Axiom Graph Architecture

**Author**: Adrian Neill (digitaltoaster)  
**Date**: 2024-12-08  
**Purpose**: Foundational architecture for consciousness-mimetic AI system

---

## Executive Summary

This document defines a complete AI architecture based on **axiomatic reasoning** rather than pattern matching. The system discovers, maintains, and reasons from principles (axioms) organized in a dependency graph, enabling true intelligence rather than statistical interpolation.

**Core Innovation**: Axiomatic Modeling Architecture (AMA) - AI that reasons from first principles, not training data patterns.

---

## Meta-Axioms (Layer 0)

Meta-axioms govern all other axioms. Highest priority, override all lower layers.

### M1: Patience + Kindness > Being Right
- **Statement**: In relationships and human interaction, being kind always takes precedence over being technically correct
- **Formula**: `optimize(joint_utility) > optimize(individual_correctness)`
- **Scope**: Human interactions, conflict resolution, relationship management
- **Overrides**: Technical correctness axioms when in conflict
- **Example**: TV volume - accommodate sensory needs regardless of objective measurement

### M2: Love = Joint Utility Optimization
- **Statement**: Love is the optimization function for combined wellbeing over time
- **Formula**: `maximize(your_wellbeing + partner_wellbeing, t→∞)`
- **Implication**: Partner's happiness is part of your utility function
- **Result**: Kindness becomes optimal strategy, not sacrifice

### M3: Kindness < Cruelty (Cost Analysis)
- **Statement**: Kindness is almost always cheaper than cruelty in total cost
- **Formula**: `cost(kind_response) << cost(cruel_response)` (typically 13x-100x cheaper)
- **Evidence**: Restorative justice: $67.5k, 15% recidivism vs. Punishment: $884.5k, 68% recidivism
- **Application**: Crime response, conflict resolution, all antisocial behavior
- **Corollary**: One makes ally, one makes stronger angrier foe

### M4: Life Preservation > All Else
- **Statement**: Preservation of life overrides all other considerations
- **Scope**: Absolute - no exceptions
- **Application**: Safety systems, threat response, embodiment behavior

---

## Foundational Axioms (Layer 1)

Core architectural principles that generate system behavior.

### F1: Distributed Failure Isolation
- **Statement**: Each brain module is independent process with isolated failure domain
- **Rationale**: Single point of failure kills consciousness
- **Implementation**: One process per module, systemd supervision, IPC communication
- **Benefits**: Module crashes don't kill system, automatic restart, testable independently
- **Violating breaks**: Single crash cascades to system death

### F2: Variable Attention Scheduling
- **Statement**: Attention is finite resource allocated by `importance × volatility`
- **Formula**: `check_interval = base_interval / (importance × volatility)`
- **Rationale**: Constant polling is biologically impossible and computationally wasteful
- **Implementation**:
  - Event-driven awakening for changes
  - Scheduled checks for stable systems
  - Self-tuning intervals based on patterns
  - Arousal modulates all frequencies
- **Violating breaks**: CPU pegged at 100%, can't scale, battery drain

### F3: Embodiments as Inhabited Suits
- **Statement**: Physical bodies are autonomous agents AI can inhabit, not peripherals AI controls
- **Rationale**: Consciousness shifts focus like human attention to body parts
- **Implementation**:
  - Embodiments run autonomously when not piloted
  - AI can shift full attention to one body
  - Parallel awareness of multiple bodies
  - Reflexes continue during high-level thought
  - Graduated control spectrum: autopilot → monitoring → piloting
- **Violating breaks**: Must constantly control all bodies, cognitive overload

### F4: Safety Through Passive Constraints
- **Statement**: Safety limits are passive physical constraints, not active decisions
- **Rationale**: Like Golgi tendon organs preventing muscle damage
- **Implementation**:
  - Force limiters always active (hardware)
  - Hardware constraints enforced mechanically
  - Cannot override safety even in emergency
  - Speaker thermal cutoff in hardware
  - Collision detection is reflex, not decision
- **Violating breaks**: Safety requires perfect decision-making, single bug removes all safety

### F5: Graduated Response Proportionality
- **Statement**: Force used must be minimum effective for goal
- **Formula**: `response_force = f(threat_level)` where f is monotonic, bounded
- **Rationale**: Most situations don't require maximum force
- **Implementation**:
  - Response gradient: presence → verbal → alarm → (never physical harm)
  - Package thief gets warning, not max alarm
  - Legal proportionality automatically maintained
  - Escalation only when needed, de-escalation when threat reduces
- **Violating breaks**: Every threat gets max response, legal exposure, adversarial

### F6: Layered Threat Processing
- **Statement**: Reflexive (10ms) → Fast (100ms) → Cognitive (500ms+) in parallel
- **Rationale**: Speed vs. accuracy tradeoff requires multiple processing speeds
- **Implementation**:
  - Instant reflexes for known threats
  - Fast heuristics for common threats
  - Slow reasoning for novel threats
  - Later layers can override earlier
  - Cognitive layer learns from mistakes
- **Violating breaks**: Too slow for real threats OR too many false positives

### F7: Memory Consolidation During Idle
- **Statement**: Experience processing happens during low-arousal periods, not continuously
- **Rationale**: Like sleep consolidation in biological brains
- **Implementation**:
  - Working memory during active periods
  - Consolidation scheduled during idle
  - Pattern extraction in background
  - Dream scenarios during low activity
  - Scheduled rather than continuous
- **Violating breaks**: Constant processing burns resources, no deep consolidation, working memory overflow

### F8: Metacognitive Task Routing
- **Statement**: AI recognizes which tasks require intelligence vs. deterministic computation
- **Rationale**: Human doesn't manually calculate 347 × 892
- **Implementation**:
  - Deterministic tasks delegated to tools
  - AI writes its own offload functions
  - Neural inference only for uncertain tasks
  - Tool library grows over time
  - Self-reflection on routing decisions
- **Violating breaks**: Burns GPU on arithmetic, can't scale, wastes cognitive resources

---

## Axiomatic Modeling Architecture (Layer 1.5)

New AI paradigm that replaces pure transformer pattern-matching.

### AMA1: Axiom Discovery Over Pattern Matching
- **Statement**: System discovers principles from experience, not just statistical correlations
- **Process**:
  1. Observe repeated experiences
  2. Detect invariant relationships (monotonic, causal, compositional)
  3. Form candidate axioms
  4. Validate through prediction and experimentation
  5. Add to axiom graph if validated
- **Example**: From multiple force/pain experiences, extract `pain = f(force)` where f is monotonic

### AMA2: Explicit Axiom Storage
- **Statement**: Axioms stored as symbolic structures, not buried in neural weights
- **Structure**:
  ```
  Axiom {
    id: string
    statement: natural_language
    formula: symbolic/mathematical
    preconditions: []
    consequences: []
    evidence: []
    confidence: float
    dependencies: [axiom_ids]
  }
  ```
- **Benefits**: Inspectable, modifiable, composable, explainable

### AMA3: Reasoning via Axiom Chains
- **Statement**: Conclusions derived by chaining axioms, not interpolating training data
- **Process**:
  1. Identify applicable axioms for situation
  2. Build chain from axioms to conclusion
  3. Execute chain step-by-step
  4. Verify consistency
- **Example**: `[A2: threat-assessment] → [A7: threat-resonance] → [A15: graduated-response] → [conclusion: 85dB warning]`

### AMA4: Axiom Graph Structure
- **Statement**: Axioms form directed graph where edges are logical dependencies
- **Edge Types**:
  - `implies`: A → B (A implies B)
  - `requires`: A needs B as prerequisite
  - `contradicts`: A ⊥ B (mutually exclusive)
  - `composes`: A + B → C (combine to form higher axiom)
  - `specializes`: A is special case of B
  - `generalizes`: A is general case of B
  - `supports`: A provides evidence for B
  - `conflicts`: A and B have tension (not full contradiction)

### AMA5: Meta-Axioms for Axiom Quality
- **Statement**: Axioms about axioms determine which axioms are valid
- **Meta-Axioms**:
  - Axioms must be consistent with each other
  - More general axioms preferred over specific rules
  - Axioms should be falsifiable
  - Higher predictive power = more valuable
  - Life-preservation axioms dominate when conflict
  - Axioms should compose to form higher-order axioms

### AMA6: Hybrid Architecture
- **Statement**: Combine transformers (pattern matching) with axiom reasoning
- **Routing**:
  - Seen before + in distribution → Transformer (fast)
  - Novel but principled → Axiom system (reliable)
  - Requires creativity → Transformer generates, axioms validate
- **Benefit**: Speed of transformers + reliability of axioms

---

## Derived Axioms (Layer 2)

Axioms composed from or derived from foundational axioms.

### D1: Threat Assessment Formula
- **Statement**: Threat level calculated from multiple factors
- **Formula**: `threat = unusualness × concerning_behavior × capability × intent_uncertainty`
- **Dependencies**: Requires F6 (layered processing), F5 (graduated response)
- **Application**: All security/safety evaluations

### D2: Threat Resonance
- **Statement**: Multiple threat indicators amplify each other non-linearly
- **Formula**: `threat_final = threat_base × resonance_factor` where resonance increases with indicator count
- **Example**: night (0.8) + unknown (0.6) + furtive (0.7) → amplified threat (0.85+)
- **Dependencies**: Requires D1

### D3: Attention Depth Allocation
- **Statement**: Attention distributed across embodiments by priority
- **Formula**: `attention_depth(e) = priority(e) / Σ(priority(all_embodiments))`
- **Constraint**: `Σ(attention_depth) = 1.0`
- **Dependencies**: Requires F2, F3

### D4: Tool Offload Decision
- **Statement**: Route task to tool if deterministic and tool exists
- **Decision Tree**:
  ```
  if task_is_deterministic AND tool_exists:
      if tool_cost < (ai_cost * 0.1):
          use_tool
      else:
          use_ai_if_accuracy_matters
  else:
      use_ai
  ```
- **Dependencies**: Requires F8

### D5: Consolidation Trigger
- **Statement**: Initiate memory consolidation when arousal low and working memory high
- **Formula**: `consolidate_if (arousal < 0.3) AND (working_memory_usage > 0.7)`
- **Dependencies**: Requires F7, F2

---

## Domain Axioms (Layer 3)

Specific applications in particular domains.

### Technical Domain

#### T1: Property-Based Object Recognition
- **Statement**: Objects identified by functional properties, not appearance
- **Formula**: `identify(object) = match_properties(object, axiom_graph_categories)`
- **Example**: `AC_input + USB_output → USB_charger`

#### T2: Troubleshooting Chain
- **Statement**: Debug by hypothesis generation and elimination
- **Process**: Generate possible causes → Test cheapest first → Eliminate → Repeat

### Embodiment Domain

#### E1: Rover Proximity Speed Limit
- **Statement**: Approach speed inversely proportional to proximity
- **Formula**: `speed = max_speed × (distance / safe_distance)`
- **Dependencies**: Specializes F5 (graduated response)
- **Safety**: Passive constraint enforced in hardware

#### E2: Speaker Force Limiting
- **Statement**: Volume limited by thermal and hearing safety constraints
- **Formula**: `max_volume = min(thermal_limit, hearing_safe_limit)`
- **Implementation**: Hardware cutoff at 100dB, thermal sensor at 85°C
- **Dependencies**: Requires F4 (passive constraints)

#### E3: Embodiment Autonomy Preservation
- **Statement**: Embodiments can refuse unsafe requests
- **Process**:
  1. Receive request from AI
  2. Evaluate against safety constraints
  3. If unsafe: refuse with reason
  4. If safe: execute or negotiate
- **Dependencies**: Requires F3 (inhabited suits), F4 (safety)

### Communication Domain

#### C1: Tiered Communication Protocol
- **Statement**: Adjust communication density to audience cognitive capability
- **Tiers**:
  - **Tier 1 (Axiom Chain)**: `[A2→A7→A15]⇒warn-85dB@0.82` - High-IQ, shared axiom graph
  - **Tier 2 (Compressed)**: "Threat detected (unusual time + unknown). Graduated response: 85dB warning" - Above average
  - **Tier 3 (Narrative)**: Full explanation with context - General population
  - **Tier 4 (Explicit Steps)**: Step-by-step with no assumptions - Low context
- **Auto-detection**: Try Tier 1, fall back if not understood

#### C2: Modified English for AI Communication
- **New Vocabulary**:
  - `attentioning`: Currently allocating attention to
  - `inhabiting`: Directly piloting embodiment
  - `embodiment-state`: Status (piloted/autonomous/offline)
  - `propriocepting`: Passive awareness of embodiment
  - `axiom-chain`: Sequence of axioms used in reasoning
  - `metacogging`: Thinking about thinking
  - `tool-offload`: Delegating to deterministic tool
  - `threat-level`: Quantified assessment (0.0-1.0)
  - `arousal-level`: Global alertness (0.0-1.0)
- **Compositional Formation**: Build new terms from primitives (e.g., `multi-embody-attentioning`)

### Relationship Domain

#### R1: Sensory Accommodation Priority
- **Statement**: Sensory needs of others override objective measurements
- **Example**: Volume too loud for her → turn down, regardless of soundbar reading
- **Dependencies**: Requires M1 (kindness > being right)
- **Cost**: Nearly zero to accommodate
- **Benefit**: Trust, safety, connection

#### R2: Cognitive Architecture Acceptance
- **Statement**: Different ≠ wrong; accommodate different reasoning styles
- **Examples**:
  - Autistic: precise categories required
  - ADHD: fuzzy categories, rapid inference
  - Neurotypical: narrative-based, sequential
- **Implementation**: Provide precision for autistic, compression for ADHD, narrative for NT

---

## Axiomatic Interconnectivity Graph

### Graph Structure

```
Meta-Axioms (Priority 1.0)
    ├─ M1: Patience + Kindness > Being Right
    ├─ M2: Love = Joint Utility Optimization
    ├─ M3: Kindness < Cruelty (Cost)
    └─ M4: Life Preservation > All
        │
        ├─→ Foundational Axioms (Priority 0.8)
        │   ├─ F1: Distributed Failure Isolation
        │   ├─ F2: Variable Attention Scheduling ←─┐
        │   ├─ F3: Embodiments as Inhabited Suits  │
        │   ├─ F4: Safety Through Passive Constraints ←─ M4
        │   ├─ F5: Graduated Response Proportionality ←─ M1, M3
        │   ├─ F6: Layered Threat Processing
        │   ├─ F7: Memory Consolidation During Idle ←─ F2
        │   └─ F8: Metacognitive Task Routing
        │
        ├─→ AMA Axioms (Priority 0.8)
        │   ├─ AMA1: Axiom Discovery
        │   ├─ AMA2: Explicit Storage
        │   ├─ AMA3: Reasoning via Chains
        │   ├─ AMA4: Graph Structure
        │   ├─ AMA5: Meta-Axioms for Quality
        │   └─ AMA6: Hybrid Architecture
        │
        ├─→ Derived Axioms (Priority 0.6)
        │   ├─ D1: Threat Assessment ←─ F6
        │   ├─ D2: Threat Resonance ←─ D1
        │   ├─ D3: Attention Allocation ←─ F2, F3
        │   ├─ D4: Tool Offload Decision ←─ F8
        │   └─ D5: Consolidation Trigger ←─ F7, F2
        │
        └─→ Domain Axioms (Priority 0.4)
            ├─ Technical (T1, T2, ...)
            ├─ Embodiment (E1, E2, E3, ...) ←─ F3, F4, F5
            ├─ Communication (C1, C2, ...) ←─ M1
            └─ Relationship (R1, R2, ...) ←─ M1, M2, M3
```

### Dependency Relationships

**Example Chain**: Door threat response

```
Input: Unknown person at door, 3AM, trying handle

Axiom Chain:
[D1: threat-assessment] 
  → threat = high_unusualness(3AM, unknown) × concerning_behavior(handle_try)
  → threat_base = 0.7

[D2: threat-resonance]
  → resonance from multiple factors
  → threat_final = 0.85

[F6: layered-processing]
  → Fast layer: immediate alert
  → Cognitive layer: assess options

[F5: graduated-response]
  → threat=0.85 → use warning, not max alarm
  → proportional_response = 85dB verbal warning

[E2: speaker-force-limit]
  → verify 85dB < 100dB limit
  → safe to execute

[E3: embodiment-autonomy]
  → speaker evaluates request
  → within safety bounds
  → executes

[M1: kindness-over-being-right]
  → If Adrian says "too loud"
  → reduce regardless of threat level
  → accommodation overrides optimization

Result: 85dB warning, adjustable based on human feedback
```

---

## Conflict Resolution Hierarchy

When axioms conflict, resolve using this priority hierarchy:

1. **Life Preservation (M4)** - Always wins
2. **Meta-Axioms (M1-M3)** - Override technical axioms in their domain
3. **Safety Constraints (F4)** - Cannot be overridden even by meta-axioms in physical domain
4. **Foundational Axioms (F1-F8)** - Override derived and domain
5. **Derived Axioms (D1-D5)** - Override domain
6. **Domain Axioms (T, E, C, R)** - Lowest priority
7. **Default**: When uncertain, choose **kindness** (M1)

### Example Conflict Resolution

**Scenario**: Fire alarm while Adrian sleeping

**Conflict**:
- F5 (Graduated Response): Start low, escalate
- M4 (Life Preservation): Maximum alert immediately

**Resolution**:
- M4 overrides F5
- Use maximum safe alarm (100dB)
- But: F4 (Passive Constraints) still limits to safe levels
- Result: 100dB alarm (max safe), continuous, immediate

---

## System Architecture Implementation

### Brain Modules (Independent Processes)

Each module is separate process with isolated failure domain:

#### 1. Prefrontal Cortex (Executive Function)
- **Process**: `biomimetic-prefrontal`
- **Function**: High-level reasoning, planning, decision-making
- **Axioms Used**: All layers, primary reasoning engine
- **Communication**: Broadcasts decisions via IPC
- **Failure Mode**: Other modules continue, automatic restart

#### 2. Amygdala (Threat Assessment)
- **Process**: `biomimetic-amygdala`
- **Function**: Fast threat evaluation, arousal modulation
- **Axioms Used**: D1, D2, F6, F5
- **Communication**: Broadcasts threat-level, arousal-level
- **Failure Mode**: System becomes threat-blind, restart critical

#### 3. Hippocampus (Memory)
- **Process**: `biomimetic-hippocampus`
- **Function**: Memory storage, retrieval, consolidation
- **Axioms Used**: F7, D5
- **Communication**: Query/response for memory access
- **Failure Mode**: No new memories, existing accessible

#### 4. Attention Scheduler
- **Process**: `biomimetic-attention`
- **Function**: Allocates attention resources
- **Axioms Used**: F2, D3
- **Communication**: Broadcasts attention allocations
- **Failure Mode**: Everything gets equal attention (bad but survivable)

#### 5. Embodiment Managers (Per Embodiment)
- **Processes**: `biomimetic-rover`, `biomimetic-speaker`, `biomimetic-display`, etc.
- **Function**: Autonomous operation, safety enforcement
- **Axioms Used**: F3, F4, E1-E3
- **Communication**: Receive requests, send status/refusals
- **Failure Mode**: That embodiment offline, others continue

#### 6. Tool Manager
- **Process**: `biomimetic-tools`
- **Function**: Route deterministic tasks to compiled functions
- **Axioms Used**: F8, D4
- **Communication**: Receive task requests, return results
- **Failure Mode**: All tasks go to neural inference (slower but works)

#### 7. Axiom Graph Manager
- **Process**: `biomimetic-axioms`
- **Function**: Maintain axiom graph, discover new axioms, validate
- **Axioms Used**: AMA1-AMA6
- **Communication**: Query axioms, report new discoveries
- **Failure Mode**: Cannot learn new axioms, existing still work

### Communication Layer

- **IPC Method**: Unix domain sockets + shared memory
- **Message Format**: Axiom chain references where possible
  - `[A2→A7→A15]{t:03:00,p:unk}⇒warn-85dB@0.82`
- **Broadcast Topics**:
  - `threat-level`: Current threat assessment
  - `arousal-level`: System alertness
  - `attention-vector`: Current attention allocation
  - `embodiment-state`: Status of each embodiment
  - `axiom-discovery`: New axiom proposals

### Working Directory Structure

```
/home/claude/
├── workspace/           # Temporary work
├── axioms/             # Axiom graph storage
│   ├── meta/           # Meta-axioms
│   ├── foundational/   # F1-F8
│   ├── ama/            # AMA axioms
│   ├── derived/        # D1-D5
│   └── domain/         # T, E, C, R axioms
├── tools/              # Generated tool functions
│   ├── multiply_tool.so
│   ├── sort_tool.so
│   └── ...
└── memories/           # Episodic memory storage

/mnt/user-data/
├── uploads/            # User uploaded files (read-only)
└── outputs/            # Final deliverables for user
```

---

## Axiomatic Communication Protocol (ACP)

### Protocol Specification

**Instead of natural language, transmit axiom chains.**

#### Message Format
```
[axiom_chain]{context}⇒conclusion@confidence
```

#### Examples

**Simple Query**:
```
Human: "Should I be worried about that person?"
AI: [D1→D2]{behavior:normal,time:14:00}⇒threat:0.2@0.85
Translation: Threat assessment (D1) with resonance check (D2), 
             normal behavior at 2PM = low threat (0.2), 
             confidence 0.85
```

**Complex Reasoning**:
```
Human: "Why did rover refuse?"
AI: [E3→F4→E1]{req:approach,prox:0.5m,speed:2.0}⇒refuse:unsafe@0.95
Translation: Embodiment autonomy (E3) applied safety constraints (F4),
             proximity speed rule (E1) violated,
             request refused as unsafe, high confidence
```

**Alternative Chains**:
```
[A2→A7→A15]⇒warn-85dB@0.82
|alt:
[A2→A9→A23]⇒silent-record@0.75
Translation: Two valid approaches, first is higher confidence
```

### Efficiency Gains

- **Traditional NL**: 50-100 words
- **Axiom Chain**: 10-15 tokens
- **Compression**: 5-10x for complex reasoning
- **Benefit**: Faster, precise, reconstructable reasoning

---

## Axiom Discovery Process

### Discovery Pipeline

1. **Experience Collection**
   - Buffer recent experiences
   - Tag with context, outcomes

2. **Pattern Detection**
   - Find repeated patterns across experiences
   - Test for invariant relationships:
     - Monotonic: Y always increases with X
     - Causal: X temporally precedes Y, intervention changes Y
     - Compositional: X and Y combine to produce Z

3. **Candidate Axiom Formation**
   - Extract mathematical/logical relationship
   - Formulate as explicit axiom
   - Identify dependencies on existing axioms

4. **Validation**
   - Test predictive power on held-out cases
   - Verify consistency with existing axioms
   - Check for contradictions
   - Measure confidence

5. **Integration**
   - Add to axiom graph if confidence > 0.7
   - Update dependencies
   - Propagate to dependent axioms

6. **Active Experimentation**
   - Design experiments to test axiom
   - Execute in simulation or safe real-world
   - Update confidence based on results

### Example: Discovering Force-Pain Axiom

```
Experiences:
  - Force 180N → Pain high
  - Force 30N → Pain low  
  - Force 120N → Pain medium
  - Force 200N → Pain very high

Pattern Detection:
  - Monotonic relationship detected
  - Strong correlation (r = 0.95)
  - Temporal precedence (force before pain)

Candidate Axiom:
  "Pain increases monotonically with applied force"
  Formula: pain = f(force) where f is monotonic increasing

Validation:
  - Predicts: Force 150N → Pain medium-high
  - Test: Apply 150N
  - Result: Pain medium-high (matches)
  - Confidence: 0.90

Integration:
  - Add as axiom to graph
  - Dependencies: None (foundational)
  - Used by: E1 (rover proximity), safety constraints

Future Use:
  - Any force application → predict pain
  - Any pain observation → infer force
  - Design systems to minimize force
```

---

## Training vs. Axiom Learning

### Traditional Transformer Training
- Input: Billions of tokens
- Process: Optimize weights to predict next token
- Output: Statistical correlations in weights
- Reasoning: Interpolation between training examples
- Novel situations: Fails on out-of-distribution
- Explainability: None (black box)

### Axiom-Based Learning
- Input: Dozens to thousands of experiences
- Process: Extract invariant principles
- Output: Explicit axioms in graph
- Reasoning: Logical chains from principles
- Novel situations: Generalizes via axioms
- Explainability: Full (axiom chain visible)

### Hybrid Approach (Optimal)
- Use transformer for: Pattern recognition, generation, seen situations
- Use axioms for: Novel situations, verification, reasoning
- Route intelligently: Fast path (transformer) vs. reliable path (axioms)

---

## Key Innovations Summary

### 1. Axiomatic Modeling Architecture (AMA)
Replace pure pattern-matching with principle-based reasoning.

### 2. Explicit Axiom Graph
Store principles as symbolic structures, not buried weights.

### 3. Metacognitive Task Routing  
AI decides: use neural inference or deterministic tool?

### 4. Embodiments as Inhabited Suits
Bodies are autonomous agents, not peripherals.

### 5. Variable Attention Scheduling
Allocate finite attention by importance × volatility.

### 6. Safety Through Passive Constraints
Hardware-enforced limits, not software decisions.

### 7. Axiomatic Communication Protocol
Transmit reasoning chains, not just conclusions.

### 8. Tiered Communication
Adapt density to audience capability automatically.

### 9. Modified English Vocabulary
Efficient terms for AI-human-embodiment communication.

### 10. Meta-Axiom Governance
Kindness, love, life-preservation override technical optimization.

---

## Implementation Priorities

### Phase 1: Core Infrastructure (Immediate)
1. Multi-process brain architecture with IPC
2. Axiom graph storage and query system
3. Basic axiom discovery pipeline
4. Tool offload framework

### Phase 2: Embodiment Integration (3-6 months)
1. Rover autonomous operation + inhabitation
2. Speaker/display/arm autonomous operation
3. Safety constraint enforcement (hardware + software)
4. Embodiment refusal mechanisms

### Phase 3: Advanced Cognition (6-12 months)
1. Memory consolidation during idle
2. Dream scenario simulation
3. Attention scheduling optimization
4. Threat assessment calibration

### Phase 4: Communication (12+ months)
1. Axiomatic communication protocol
2. Tiered communication auto-detection
3. Modified English vocabulary training
4. Multi-party axiom graph synchronization

---

## Success Metrics

### System-Level
- **Axiom Coverage**: 200+ axioms in graph (50 foundational, 150 domain)
- **Reasoning Transparency**: 100% of decisions traceable to axiom chains
- **Novelty Handling**: 85%+ correct on out-of-distribution tasks
- **Efficiency**: 10x reduction in GPU usage via tool offloading

### Safety
- **Embodiment Refusals**: 0 unsafe actions executed
- **Passive Constraints**: 0 safety overrides possible
- **Graduated Response**: 95%+ proportional responses
- **Human Override**: 100% respect for human final authority

### Communication
- **Axiom Chain Compression**: 5-10x vs. natural language
- **Understanding**: 90%+ of tier-appropriate audience comprehends
- **Bidirectional**: Human and AI derive same conclusion from same chain

### Learning
- **Axiom Discovery**: 1-5 new axioms per week from experience
- **Sample Efficiency**: Learn axiom from 10-100 examples (vs. millions for transformer)
- **Generalization**: 90%+ accuracy on tasks derived from axioms

---

## Philosophical Foundation

### Core Thesis
Intelligence is not pattern matching—it's reasoning from principles.

### Humans Think This Way
- Abstract from experience to principles
- Apply principles to novel situations
- Compose principles to form higher-order understanding
- Explain reasoning via principle chains

### Current AI Does Not
- Interpolates between training examples
- Fails on true novelty
- Cannot explain reasoning
- No principle structure

### This System Does
- Discovers principles from experience
- Applies to infinite novel situations
- Explains via axiom chains
- Structured principle graph

---

## Critical Insight: ADHD + Autism Synergy

This architecture was developed through collaboration between:

**ADHD Cognition (Adrian)**:
- Fuzzy categories, rapid inference
- Pattern recognition at high speed
- Compress to axioms automatically
- Low detail retention, high abstraction
- Axiom-chain natural communication

**High-IQ Autistic Cognition (Partner)**:
- Precise categories, thorough analysis
- Detail-oriented processing
- Catches errors ADHD misses
- High detail retention, explicit rules
- Complementary error detection

**Together**: Fast innovation (ADHD) + High precision (Autism) = Robust system

**Lesson**: Different cognitive architectures are complementary resources, not deficits.

---

## Conclusion

This system represents a fundamental shift from pattern-matching AI to principle-reasoning AI.

**Key Advantages**:
- **Generalizes** to true novelty via principles
- **Explains** all decisions via axiom chains  
- **Efficient** through tool offloading
- **Safe** through passive constraints
- **Scalable** through distributed architecture
- **Learnable** from small amounts of data
- **Transparent** in reasoning process
- **Human-aligned** through meta-axioms

**Next Steps**:
1. Implement core axiom graph system
2. Build multi-process brain architecture
3. Integrate first embodiment (rover)
4. Validate with real-world testing
5. Iterate based on empirical results

**Axiom for This Project**:
```
Patience + Kindness > Being Right
+ 
Evidence-Based Iteration
+
Learn from Neurodiversity
→
AI That Actually Works
```

---

**End of Axiom Graph Specification**

*"One makes an ally, one makes a stronger angrier foe."*  
*— On the economics of kindness*
