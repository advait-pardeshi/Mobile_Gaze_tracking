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
    static let fallback = ["please", "thanks", "now", "today",
                           "more", "help", "stop"]

    func candidates(for question: PromptQuestion,
                    prefix: [String],
                    count: Int) -> [String] {
        let offered = question.node(for: prefix) ?? Self.fallback
        return Array(offered.prefix(count))
    }
}

/// The fixed question set. Order is shuffled per run; the set itself is
/// constant across participants so the trials are comparable.
enum PromptQuestionSet {

    /// Five questions, ~3–5 selections each.
    ///
    /// `drink` is deliberately cued with **"I want to drink water"** — the
    /// exact sentence Experiment 3 composes from its static 4×3 grid. That
    /// one trial is a like-for-like comparison of static vs. predictive
    /// selection on identical words with an identical dwell time, which is
    /// the cleanest number this experiment can produce.
    static let standard: [PromptQuestion] = [
        feeling, drink, hungry, help, going,
    ]

    static let feeling = PromptQuestion(
        key: "feeling",
        text: "How are you feeling today?",
        cuedAnswer: ["I", "feel", "good", "today"],
        tree: [
            "": ["I", "I'm", "Not", "Yes", "No", "Very", "Thanks"],
            "I": ["feel", "am", "want", "need", "don't", "was", "have"],
            "I feel": ["good", "bad", "tired", "okay", "happy", "sick", "better"],
            "I feel good": ["today", "now", "thanks", "really", "very", "much", "please"],
            "I feel bad": ["today", "now", "really", "very", "much", "sorry", "please"],
            "I feel tired": ["today", "now", "really", "very", "much", "always", "please"],
            "I'm": ["good", "bad", "tired", "okay", "happy", "sick", "fine"],
            "Not": ["good", "bad", "well", "very", "really", "today", "much"],
        ])

    static let drink = PromptQuestion(
        key: "drink",
        text: "What would you like to drink?",
        cuedAnswer: ["I", "want", "to", "drink", "water"],
        tree: [
            "": ["I", "I'd", "Yes", "No", "Water", "Please", "Nothing"],
            "I": ["want", "would", "need", "like", "don't", "feel", "am"],
            "I want": ["to", "water", "more", "some", "a", "juice", "tea"],
            "I want to": ["drink", "eat", "go", "sleep", "have", "stop", "help"],
            "I want to drink": ["water", "juice", "milk", "tea", "coffee", "more", "please"],
            "I want water": ["please", "now", "thanks", "more", "today", "cold", "help"],
            "I need": ["to", "water", "help", "more", "some", "a", "food"],
        ])

    static let hungry = PromptQuestion(
        key: "hungry",
        text: "Are you hungry?",
        cuedAnswer: ["Yes", "I", "want", "to", "eat"],
        tree: [
            "": ["Yes", "No", "I", "I'm", "Not", "Maybe", "Please"],
            "Yes": ["I", "please", "thanks", "very", "a", "now", "I'm"],
            "Yes I": ["want", "am", "need", "would", "feel", "don't", "can"],
            "Yes I want": ["to", "food", "more", "some", "water", "a", "please"],
            "Yes I want to": ["eat", "drink", "go", "sleep", "have", "stop", "help"],
            "No": ["I", "thanks", "not", "I'm", "please", "now", "thank"],
            "No I": ["am", "don't", "feel", "want", "need", "was", "can't"],
        ])

    static let help = PromptQuestion(
        key: "help",
        text: "Do you need any help?",
        cuedAnswer: ["No", "thank", "you"],
        tree: [
            "": ["No", "Yes", "I", "Not", "Please", "Maybe", "Thanks"],
            "No": ["thank", "thanks", "I'm", "not", "please", "I", "now"],
            "No thank": ["you"],
            "Yes": ["please", "I", "help", "thanks", "now", "a", "very"],
            "Yes please": ["help", "now", "thanks", "more", "I", "today", "stop"],
            "I": ["need", "want", "am", "don't", "feel", "can't", "would"],
            "I need": ["help", "you", "more", "to", "water", "a", "please"],
        ])

    static let going = PromptQuestion(
        key: "going",
        text: "Where do you want to go?",
        cuedAnswer: ["I", "want", "to", "go", "home"],
        tree: [
            "": ["I", "I'd", "Home", "Outside", "Nowhere", "Please", "To"],
            "I": ["want", "would", "need", "like", "don't", "am", "feel"],
            "I want": ["to", "home", "out", "a", "some", "more", "help"],
            "I want to": ["go", "drink", "eat", "sleep", "stay", "stop", "help"],
            "I want to go": ["home", "outside", "out", "now", "bed", "please", "back"],
            "I want to stay": ["here", "home", "now", "please", "today", "more", "inside"],
            "Home": ["please", "now", "today", "thanks", "more", "help", "stop"],
        ])
}
