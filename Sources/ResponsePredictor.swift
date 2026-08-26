import Foundation

/// Next-word prediction for Experiment 4 (prompted predictive communication).
///
/// The grid in Experiment 4 shows only seven words at a time, so what the
/// participant can say is decided entirely by what this returns. That makes
/// the predictor part of the *apparatus*, not part of the result — and an
/// apparatus that behaves differently for each participant cannot be compared
/// across them.
///
/// Hence the protocol, and hence the fact that the shipped conformance is a
/// hand-authored trie rather than a language model:
///
///  * **Deterministic.** Every participant sees the identical grid at the
///    identical step of the identical question. A sampled LM would make the
///    words-per-minute comparison against Experiment 3 a comparison of two
///    interfaces *and* two draws from a distribution.
///  * **Zero latency.** Inference time inside `refreshGrid()` would land
///    directly in `since_prev_s`, i.e. in the headline rate metric.
///  * **Reachable by construction.** In the cued condition the answer the
///    participant was told to give must be on the grid at every step or the
///    trial is unwinnable. A trie can be *checked* for that before the run
///    starts (`PromptQuestion.unreachableCuedStep`); a model can only be
///    patched afterwards by injecting the missing word, which is precisely
///    the bias the experiment is trying to measure.
///
/// An n-gram table or an on-device LM can be added later as a second
/// conformance and a condition flag, with the trie as the reference they are
/// compared against. Nothing in `PredictiveTaskController` assumes the trie.
protocol ResponsePredictor {
    /// Candidates for the next word, best-first, given the question being
    /// answered and the words chosen so far.
    ///
    /// May return fewer than `count`; the overlay renders the leftover cells
    /// as inert blanks rather than padding with filler, because a filler word
    /// would be selectable and would show up as a false selection that the
    /// predictor, not the participant, caused.
    func candidates(for question: PromptQuestion,
                    prefix: [String],
                    count: Int) -> [String]
}

/// One prompted question and the response space it opens up.
struct PromptQuestion: Identifiable {
    /// Stable key, used in logs and filenames.
    let key: String
    /// Spoken and shown verbatim during the prompt phase.
    let text: String
    /// The answer the participant is told to give in the cued condition.
    /// Ignored in the free condition.
    let cuedAnswer: [String]
    /// Prefix → candidates, best-first. The key is the chosen words joined
    /// with a single space; `""` is the sentence-initial state.
    let tree: [String: [String]]

    var id: String { key }

    /// Candidates for `prefix`, or nil when the trie has no node for it.
    func node(for prefix: [String]) -> [String]? {
        tree[prefix.joined(separator: " ")]
    }

    /// Index of the first step of `cuedAnswer` whose word is not offered by
    /// the trie, or nil when the whole answer is reachable.
    ///
    /// `PredictiveTaskController.start()` calls this for every question and
    /// refuses to run if any answer is unreachable. A trial that cannot be
    /// completed is not a hard trial, it is a broken one, and it would enter
    /// the aggregate as a timeout indistinguishable from a participant who
    /// simply could not hit the cells.
    func unreachableCuedStep(count: Int) -> Int? {
        for i in cuedAnswer.indices {
            let prefix = Array(cuedAnswer.prefix(i))
            let offered = node(for: prefix) ?? ResponseTrie.fallback
            if !offered.prefix(count).contains(cuedAnswer[i]) { return i }
        }
        return nil
    }
}

/// The shipped predictor: a per-question trie of hand-authored responses.
struct ResponseTrie: ResponsePredictor {

    /// Offered when a prefix has no node — i.e. the participant has gone off
    /// the authored paths, which the free condition allows. Deliberately a
    /// set of sentence-enders so a wandering response can still be finished
    /// rather than dead-ending on a grid of blanks.
    ///
    /// These are the *only* words in the file chosen without regard to the
    /// question, and that is the point: they exist to terminate an
    /// off-script response, not to answer anything. Every word inside a
    /// question's own trie is on-topic for that question.
    static let fallback = ["please", "thanks", "now", "today",
                           "yes", "no", "okay"]

    func candidates(for question: PromptQuestion,
                    prefix: [String],
                    count: Int) -> [String] {
        let offered = question.node(for: prefix) ?? Self.fallback
        return Array(offered.prefix(count))
    }
}

/// The fixed question set. Order is shuffled per run; the set itself is
/// constant across participants so the trials are comparable.
///
/// **Authoring rule for every node below.** A candidate has to be both a
/// grammatical continuation of its prefix *and* a plausible answer to that
/// question. Nothing is padded to fill the grid: a node with four sensible
/// continuations offers four, and the overlay renders the remaining slots as
/// inert blanks (`PredictiveCellKind.blank`). That is deliberate — a filler
/// word is selectable, so padding "I want to" out to seven with `sleep`,
/// `stop` and `help` on a question about drinks manufactures selections the
/// predictor caused and scores them against the participant, and it inflates
/// the keystroke-savings figure with words nobody would ever pick.
///
/// The trie is also wide enough that the free condition rarely falls through
/// to `ResponseTrie.fallback`: each question authors the branches a
/// participant actually takes (yes/no openers, `I'm …`, `I don't …`, a bare
/// one-word answer), not just the cued path.
enum PromptQuestionSet {

    /// The full authored pool, ~3–5 selections each. A run uses a subset of
    /// this — see `session(count:)`.
    ///
    /// `drink` is deliberately cued with **"I want to drink water"** — the
    /// exact sentence Experiment 3 composes from its static 4×3 grid. That
    /// one trial is a like-for-like comparison of static vs. predictive
    /// selection on identical words with an identical dwell time, which is
    /// the cleanest number this experiment can produce.
    static let standard: [PromptQuestion] = [
        feeling, drink, hungry, help, going,
    ]

    /// The question every session must include, whatever else is drawn.
    ///
    /// Without pinning it, a run that happened not to draw `drink` would
    /// carry no comparison against Experiment 3 at all — and that comparison
    /// is the reason Experiment 4 exists in its current form.
    static let pinned = drink

    /// Number of questions a run asks. Below the pool size, so each session
    /// samples it.
    static let sessionQuestionCount = 3

    /// The questions for one run: `pinned`, plus enough others drawn at
    /// random to reach `count`.
    ///
    /// **Note this weakens cross-participant comparison**, and deliberately:
    /// the pool is authored to be constant so every participant meets the
    /// identical trial, and drawing a subset breaks that for every question
    /// except `pinned`. Per-question aggregates therefore have to be filtered
    /// by `question_key` (present on every logged row) and will have unequal
    /// n across the pool. `pinned` is the only one guaranteed a full n.
    static func session(count: Int = sessionQuestionCount) -> [PromptQuestion] {
        guard count < standard.count else { return standard }
        let rest = standard.filter { $0.key != pinned.key }
        return ([pinned] + rest.shuffled().prefix(max(0, count - 1)))
    }

    static let feeling = PromptQuestion(
        key: "feeling",
        text: "How are you feeling today?",
        cuedAnswer: ["I", "feel", "good", "today"],
        tree: [
            "": ["I", "I'm", "Not", "Very", "Good", "Tired", "Okay"],
            "I": ["feel", "am", "was", "don't", "still", "really"],
            "I feel": ["good", "bad", "tired", "okay", "happy", "sick", "better"],
            "I feel good": ["today", "now", "thanks", "too"],
            "I feel bad": ["today", "now", "sorry"],
            "I feel tired": ["today", "now", "always"],
            "I feel okay": ["today", "now", "thanks"],
            "I feel happy": ["today", "now", "thanks"],
            "I feel sick": ["today", "now", "sorry"],
            "I feel better": ["today", "now", "thanks"],
            "I am": ["good", "bad", "tired", "okay", "happy", "sick", "fine"],
            "I don't": ["feel", "know"],
            "I don't feel": ["good", "well", "bad", "okay"],
            "I'm": ["good", "bad", "tired", "okay", "happy", "sick", "fine"],
            "I'm good": ["today", "now", "thanks"],
            "I'm tired": ["today", "now", "always"],
            "I'm okay": ["today", "now", "thanks"],
            "Not": ["good", "bad", "well", "really", "today"],
            "Not good": ["today", "now", "sorry"],
            "Very": ["good", "bad", "tired", "happy", "sick"],
            "Good": ["thanks", "today", "now"],
            "Tired": ["today", "now"],
            "Okay": ["thanks", "today", "now"],
        ])

    static let drink = PromptQuestion(
        key: "drink",
        text: "What would you like to drink?",
        cuedAnswer: ["I", "want", "to", "drink", "water"],
        tree: [
            "": ["I", "I'd", "Water", "Juice", "Nothing", "Some", "Please"],
            "I": ["want", "would", "need", "like", "don't"],
            "I want": ["to", "water", "juice", "milk", "tea", "some", "more"],
            "I want to": ["drink", "have", "try"],
            "I want to drink": ["water", "juice", "milk", "tea", "coffee", "something"],
            "I want to drink water": ["please", "now", "thanks"],
            "I want water": ["please", "now", "thanks"],
            "I want juice": ["please", "now", "thanks"],
            "I want milk": ["please", "now", "thanks"],
            "I want tea": ["please", "now", "thanks"],
            "I want some": ["water", "juice", "milk", "tea", "coffee"],
            "I need": ["water", "a", "to", "some"],
            "I need water": ["please", "now", "thanks"],
            "I don't": ["want", "need"],
            "I don't want": ["anything", "water", "juice", "milk"],
            "I'd": ["like", "love", "prefer"],
            "I'd like": ["water", "juice", "milk", "tea", "coffee", "some", "a"],
            "I'd like some": ["water", "juice", "milk", "tea", "coffee"],
            "Water": ["please", "now", "thanks"],
            "Juice": ["please", "now", "thanks"],
            "Nothing": ["thanks", "please", "now"],
            "Some": ["water", "juice", "milk", "tea", "coffee"],
            "Please": ["water", "juice", "milk", "tea"],
        ])

    static let hungry = PromptQuestion(
        key: "hungry",
        text: "Are you hungry?",
        cuedAnswer: ["Yes", "I", "want", "to", "eat"],
        tree: [
            "": ["Yes", "No", "I", "I'm", "Not", "Maybe", "A"],
            "Yes": ["I", "please", "very", "a", "thanks", "now"],
            "Yes I": ["want", "am", "need", "would", "could"],
            "Yes I want": ["to", "food", "something", "some", "more"],
            "Yes I want to": ["eat", "have", "try"],
            "Yes I want to eat": ["now", "please", "something", "more"],
            "Yes I am": ["hungry", "very", "really", "now"],
            "Yes very": ["hungry", "much"],
            "Yes a": ["little", "lot", "bit"],
            "No": ["thanks", "thank", "I", "not", "I'm"],
            "No thank": ["you"],
            "No I": ["am", "don't", "just", "already"],
            "No I am": ["not", "okay", "fine", "full"],
            "No I don't": ["want", "need", "think"],
            "No I already": ["ate"],
            "Not": ["hungry", "really", "now", "very"],
            "Not hungry": ["now", "thanks", "today"],
            "Maybe": ["a", "later", "yes", "I"],
            "Maybe a": ["little", "lot", "bit"],
            "A": ["little", "lot", "bit"],
            "A little": ["hungry", "yes", "now", "please"],
            "I": ["want", "am", "need", "don't", "already"],
            "I want": ["to", "food", "something", "some", "more"],
            "I want to": ["eat", "have", "try"],
            "I'm": ["hungry", "not", "full", "okay", "very", "starving"],
            "I'm hungry": ["now", "please", "thanks"],
            "I'm not": ["hungry", "really", "now"],
        ])

    static let help = PromptQuestion(
        key: "help",
        text: "Do you need any help?",
        cuedAnswer: ["No", "thank", "you"],
        tree: [
            "": ["No", "Yes", "I", "Not", "Please", "Maybe", "Thanks"],
            "No": ["thank", "thanks", "I'm", "I", "not"],
            "No thank": ["you"],
            "No thank you": ["I'm", "now", "thanks"],
            "No thanks": ["I'm", "I", "now"],
            "No I'm": ["okay", "fine", "good", "alright"],
            "No I": ["am", "don't", "can"],
            "No I don't": ["need", "want", "think"],
            "Yes": ["please", "I", "help", "thanks"],
            "Yes please": ["help", "now", "thank", "I"],
            "Yes please help": ["me", "now", "please"],
            "Yes I": ["need", "want", "do", "can't"],
            "Yes I need": ["help", "you", "a", "some"],
            "I": ["need", "want", "am", "don't", "can't"],
            "I need": ["help", "you", "a", "some", "to"],
            "I need help": ["please", "now", "thanks"],
            "I need you": ["please", "now", "here"],
            "I can't": ["do", "reach", "move"],
            "I don't": ["need", "want", "think"],
            "I don't need": ["help", "anything", "it"],
            "Not": ["now", "really", "yet", "today"],
            "Not now": ["thanks", "please", "later"],
            "Maybe": ["later", "yes", "no", "please"],
            "Please": ["help", "come", "wait"],
            "Please help": ["me", "now", "please"],
            "Thanks": ["I'm", "I", "no"],
        ])

    static let going = PromptQuestion(
        key: "going",
        text: "Where do you want to go?",
        cuedAnswer: ["I", "want", "to", "go", "home"],
        tree: [
            "": ["I", "I'd", "Home", "Outside", "Bed", "Nowhere", "Please"],
            "I": ["want", "would", "need", "like", "don't"],
            "I want": ["to", "home", "out", "outside"],
            "I want to": ["go", "stay", "leave", "come"],
            "I want to go": ["home", "outside", "out", "back", "bed", "there", "now"],
            "I want to go home": ["now", "please", "today", "thanks"],
            "I want to go outside": ["now", "please", "today"],
            "I want to go back": ["home", "inside", "now"],
            "I want to go bed": ["now", "please"],
            "I want to stay": ["here", "home", "inside", "now"],
            "I want to leave": ["now", "please", "today"],
            "I want home": ["now", "please", "today"],
            "I need": ["to", "home", "a"],
            "I need to": ["go", "stay", "leave"],
            "I don't": ["want", "know", "care"],
            "I don't want": ["to", "anywhere"],
            "I'd": ["like", "love", "rather"],
            "I'd like": ["to", "home", "outside"],
            "I'd like to": ["go", "stay", "leave"],
            "Home": ["please", "now", "today", "thanks"],
            "Outside": ["please", "now", "today"],
            "Bed": ["please", "now"],
            "Nowhere": ["thanks", "now", "today"],
            "Please": ["home", "now", "take"],
            "Please take": ["me"],
            "Please take me": ["home", "outside", "there"],
        ])
}
