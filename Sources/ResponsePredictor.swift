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
    /// Offered when the prefix reaches a **complete** response — one whose
    /// last word ends the sentence, so the trie authors no continuation.
    ///
    /// Per-question rather than global. The generic list this replaced
    /// (`please / thanks / now / today / yes / no / okay`) was the single
    /// largest source of nonsense on the grid: any prefix the trie did not
    /// author fell to it, so a participant who picked `I would` on the drink
    /// question was offered `yes no okay` — words that neither continue the
    /// sentence nor answer the question. Closers are only ever reached from
    /// a finished sentence now (see the closure invariant on `tree` below),
    /// and each question's set is on-topic for it.
    let closers: [String]

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
            let offered = node(for: prefix) ?? closers
            if !offered.prefix(count).contains(cuedAnswer[i]) { return i }
        }
        return nil
    }

    /// **The closure invariant.** Every prefix a participant can build by
    /// picking offered words, that is *not* already a complete sentence,
    /// must have an authored node.
    ///
    /// Returns the prefixes that violate it — reachable, unfinished, and
    /// unauthored, so the grid would fall back to `closers` mid-sentence and
    /// offer words unrelated to what was picked. Empty is the only correct
    /// value; `PredictiveTaskController.start()` refuses to run otherwise.
    ///
    /// "Complete" is decided by the last word being in `PromptQuestionSet
    /// .sentenceFinal`: those are the words a response can legitimately stop
    /// on, and stopping on one is what `closers` exists to serve.
    func danglingPrefixes(count: Int) -> [String] {
        var bad: [String] = []
        var seen = Set<String>()
        var frontier: [[String]] = [[]]
        // Deepest authored path is 6 words; the bound only stops a cycle
        // introduced by a future edit from hanging the check.
        for _ in 0..<12 {
            guard !frontier.isEmpty else { break }
            var next: [[String]] = []
            for prefix in frontier {
                guard let offered = node(for: prefix) else { continue }
                for word in offered.prefix(count) {
                    let child = prefix + [word]
                    let joined = child.joined(separator: " ")
                    guard seen.insert(joined).inserted else { continue }
                    if node(for: child) != nil { next.append(child) }
                    else if !PromptQuestionSet.sentenceFinal.contains(word) {
                        bad.append(joined)
                    }
                }
            }
            frontier = next
        }
        return bad.sorted()
    }
}

/// The shipped predictor: a per-question trie of hand-authored responses.
struct ResponseTrie: ResponsePredictor {

    func candidates(for question: PromptQuestion,
                    prefix: [String],
                    count: Int) -> [String] {
        let offered = question.node(for: prefix) ?? question.closers
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
/// **And every node must be closed** (`PromptQuestion.danglingPrefixes`): if
/// a node offers a word that does not finish the sentence, the prefix ending
/// in that word needs its own node. Offering `I would` without authoring
/// `I would` is what made the grid go off-topic mid-sentence — the branch
/// dead-ended into generic closers. Where closing a branch would need a node
/// nobody would ever walk, the fix is to *not offer the word*, not to author
/// a filler node: several nodes below are deliberately three words wide for
/// that reason.
enum PromptQuestionSet {

    /// Words a response may legitimately stop on. A prefix ending in one of
    /// these is complete, so it draws `closers` rather than an authored node
    /// — that is the intended use of the fallback, not an escape hatch.
    ///
    /// Membership is a claim about the word in *sentence-final position*
    /// across this question set, which is why concrete nouns are here
    /// ("water" ends "I want to drink water") and bare modals and
    /// determiners are not ("would", "a").
    static let sentenceFinal: Set<String> = [
        // Polite / temporal enders.
        "please", "thanks", "thank", "you", "now", "today", "too", "sorry",
        "always", "later", "much", "yet", "so", "ate", "it", "manage",
        // States.
        "good", "bad", "tired", "okay", "happy", "sick", "better", "well",
        "fine", "alright", "hungry", "full", "starving",
        // Objects and quantities.
        "water", "juice", "milk", "tea", "coffee", "something", "anything",
        "food", "more", "drink", "little", "lot", "bit", "help", "me",
        // Places.
        "home", "outside", "here", "inside", "there", "back", "anywhere",
    ]

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
            "I was": ["good", "bad", "tired", "okay", "happy", "sick", "fine"],
            "I still": ["feel", "am"],
            "I still feel": ["good", "bad", "tired", "okay", "happy", "sick", "better"],
            "I still am": ["tired", "sick", "okay", "good", "happy"],
            "I really": ["feel", "am"],
            "I really feel": ["good", "bad", "tired", "okay", "happy", "sick", "better"],
            "I really am": ["tired", "sick", "okay", "good", "happy"],
            "I don't": ["feel", "know"],
            "I don't feel": ["good", "well", "bad", "okay"],
            "I don't know": ["today", "now", "sorry"],
            "I'm": ["good", "bad", "tired", "okay", "happy", "sick", "fine"],
            "I'm good": ["today", "now", "thanks"],
            "I'm bad": ["today", "now", "sorry"],
            "I'm tired": ["today", "now", "always"],
            "I'm okay": ["today", "now", "thanks"],
            "I'm happy": ["today", "now", "thanks"],
            "I'm sick": ["today", "now", "sorry"],
            "I'm fine": ["today", "now", "thanks"],
            "Not": ["good", "bad", "well", "really", "today"],
            "Not good": ["today", "now", "sorry"],
            "Not bad": ["today", "now", "thanks"],
            "Not well": ["today", "now", "sorry"],
            "Not really": ["today", "now", "thanks"],
            "Very": ["good", "bad", "tired", "happy", "sick"],
            "Very good": ["today", "now", "thanks"],
            "Very bad": ["today", "now", "sorry"],
            "Very tired": ["today", "now", "always"],
            "Very happy": ["today", "now", "thanks"],
            "Very sick": ["today", "now", "sorry"],
            "Good": ["thanks", "today", "now"],
            "Tired": ["today", "now"],
            "Okay": ["thanks", "today", "now"],
        ],
        closers: ["today", "now", "thanks", "too"])

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
            "I want to have": ["water", "juice", "milk", "tea", "coffee", "something"],
            "I want to try": ["juice", "tea", "coffee", "something"],
            "I want water": ["please", "now", "thanks"],
            "I want juice": ["please", "now", "thanks"],
            "I want milk": ["please", "now", "thanks"],
            "I want tea": ["please", "now", "thanks"],
            "I want some": ["water", "juice", "milk", "tea", "coffee"],
            "I want more": ["water", "juice", "milk", "tea", "please"],
            "I would": ["like", "love", "prefer"],
            "I would like": ["water", "juice", "milk", "tea", "coffee", "some", "a"],
            "I would like some": ["water", "juice", "milk", "tea", "coffee"],
            "I would like a": ["drink"],
            "I would like a drink": ["please", "now", "thanks"],
            "I would love": ["water", "juice", "milk", "tea", "coffee", "some"],
            "I would love some": ["water", "juice", "milk", "tea", "coffee"],
            "I would prefer": ["water", "juice", "milk", "tea", "coffee"],
            "I need": ["water", "juice", "milk", "tea", "a", "to", "some"],
            "I need water": ["please", "now", "thanks"],
            "I need juice": ["please", "now", "thanks"],
            "I need milk": ["please", "now", "thanks"],
            "I need tea": ["please", "now", "thanks"],
            "I need a": ["drink"],
            "I need a drink": ["please", "now", "thanks"],
            "I need to": ["drink"],
            "I need to drink": ["water", "juice", "milk", "tea", "coffee", "something"],
            "I need some": ["water", "juice", "milk", "tea", "coffee"],
            "I like": ["water", "juice", "milk", "tea", "coffee"],
            "I don't": ["want", "need"],
            "I don't want": ["anything", "water", "juice", "milk"],
            "I don't need": ["anything", "water", "juice", "milk"],
            "I'd": ["like", "love", "prefer"],
            "I'd like": ["water", "juice", "milk", "tea", "coffee", "some", "a"],
            "I'd like some": ["water", "juice", "milk", "tea", "coffee"],
            "I'd like a": ["drink"],
            "I'd like a drink": ["please", "now", "thanks"],
            "I'd love": ["water", "juice", "milk", "tea", "coffee", "some"],
            "I'd love some": ["water", "juice", "milk", "tea", "coffee"],
            "I'd prefer": ["water", "juice", "milk", "tea", "coffee"],
            "Water": ["please", "now", "thanks"],
            "Juice": ["please", "now", "thanks"],
            "Nothing": ["thanks", "please", "now"],
            "Some": ["water", "juice", "milk", "tea", "coffee"],
            "Please": ["water", "juice", "milk", "tea"],
        ],
        closers: ["please", "now", "thanks", "too"])

    static let hungry = PromptQuestion(
        key: "hungry",
        text: "Are you hungry?",
        cuedAnswer: ["Yes", "I", "want", "to", "eat"],
        tree: [
            "": ["Yes", "No", "I", "I'm", "Not", "Maybe", "A"],
            "Yes": ["I", "please", "very", "a", "thanks", "now"],
            "Yes I": ["want", "am", "need", "could"],
            "Yes I want": ["to", "food", "something", "some", "more"],
            "Yes I want to": ["eat", "have", "try"],
            "Yes I want to eat": ["now", "please", "something", "more"],
            "Yes I want to have": ["food", "something", "more"],
            "Yes I want to try": ["food", "something"],
            "Yes I want some": ["food", "more", "something"],
            "Yes I am": ["hungry", "very", "really", "now"],
            "Yes I am very": ["hungry", "now", "please"],
            "Yes I am really": ["hungry", "now", "please"],
            "Yes I need": ["food", "to", "some", "more"],
            "Yes I need to": ["eat"],
            "Yes I need to eat": ["now", "please", "something"],
            "Yes I need some": ["food", "more", "something"],
            "Yes I could": ["eat"],
            "Yes I could eat": ["now", "please", "something", "more"],
            "Yes very": ["hungry", "much"],
            "Yes a": ["little", "lot", "bit"],
            "No": ["thanks", "thank", "I", "not", "I'm"],
            "No thank": ["you"],
            "No I": ["am", "don't", "just", "already"],
            "No I am": ["not", "okay", "fine", "full"],
            "No I am not": ["hungry", "now", "today"],
            "No I don't": ["want", "need", "think"],
            "No I don't want": ["anything", "food", "more"],
            "No I don't need": ["anything", "food", "more"],
            "No I don't think": ["so"],
            "No I just": ["ate"],
            "No I already": ["ate"],
            "No I'm": ["full", "okay", "fine", "not"],
            "No I'm not": ["hungry", "now", "today"],
            "No not": ["hungry", "now", "today"],
            "Not": ["hungry", "really", "now", "very"],
            "Not hungry": ["now", "thanks", "today"],
            "Not really": ["thanks", "now", "today"],
            "Not very": ["hungry", "much"],
            "Maybe": ["a", "later", "yes", "I'm"],
            "Maybe a": ["little", "lot", "bit"],
            "Maybe yes": ["please", "now", "thanks"],
            "Maybe I'm": ["hungry", "full", "okay"],
            "A": ["little", "lot", "bit"],
            "A little": ["hungry", "now", "please", "thanks"],
            "I": ["want", "am", "need", "don't", "already"],
            "I want": ["to", "food", "something", "some", "more"],
            "I want to": ["eat", "have", "try"],
            "I want to eat": ["now", "please", "something", "more"],
            "I want to have": ["food", "something", "more"],
            "I want to try": ["food", "something"],
            "I want some": ["food", "more", "something"],
            "I am": ["hungry", "not", "full", "okay", "very", "starving"],
            "I am not": ["hungry", "now", "today"],
            "I am very": ["hungry", "full"],
            "I am starving": ["now", "please", "thanks"],
            "I need": ["food", "to", "some", "more"],
            "I need to": ["eat"],
            "I need to eat": ["now", "please", "something", "more"],
            "I need some": ["food", "more", "something"],
            "I don't": ["want", "need", "think"],
            "I don't want": ["anything", "food", "more"],
            "I don't need": ["anything", "food", "more"],
            "I don't think": ["so"],
            "I already": ["ate"],
            "I'm": ["hungry", "not", "full", "okay", "very", "starving"],
            "I'm hungry": ["now", "please", "thanks"],
            "I'm not": ["hungry", "really", "now"],
            "I'm not really": ["hungry", "now"],
            "I'm very": ["hungry", "full"],
            "I'm starving": ["now", "please", "thanks"],
        ],
        closers: ["now", "please", "thanks", "today"])

    static let help = PromptQuestion(
        key: "help",
        text: "Do you need any help?",
        cuedAnswer: ["No", "thank", "you"],
        tree: [
            "": ["No", "Yes", "I", "Not", "Please", "Maybe", "Thanks"],
            "No": ["thank", "thanks", "I'm", "I", "not"],
            "No thank": ["you"],
            "No thank you": ["now", "thanks", "please"],
            "No thanks": ["now", "please", "today"],
            "No I'm": ["okay", "fine", "good", "alright"],
            "No I": ["am", "don't", "can"],
            "No I am": ["okay", "fine", "good", "alright"],
            "No I don't": ["need", "want", "think"],
            "No I don't need": ["help", "anything", "it"],
            "No I don't want": ["help", "anything", "it"],
            "No I don't think": ["so"],
            "No I can": ["manage"],
            "No not": ["now", "yet", "today", "thanks"],
            "Yes": ["please", "I", "help", "thanks"],
            "Yes please": ["help", "now", "thanks", "today"],
            "Yes please help": ["me", "now", "please"],
            "Yes help": ["me", "please", "now"],
            "Yes thanks": ["now", "please", "today"],
            "Yes I": ["need", "want", "do", "can't"],
            "Yes I need": ["help", "you", "some"],
            "Yes I need help": ["please", "now", "thanks"],
            "Yes I need you": ["please", "now", "here"],
            "Yes I need some": ["help"],
            "Yes I want": ["help", "you"],
            "Yes I want help": ["please", "now", "thanks"],
            "Yes I want you": ["please", "now", "here"],
            "Yes I do": ["please", "thanks", "now"],
            "Yes I can't": ["do", "reach", "move"],
            "Yes I can't do": ["it"],
            "Yes I can't reach": ["it"],
            "Yes I can't move": ["now", "please", "thanks"],
            "I": ["need", "want", "am", "don't", "can't"],
            "I need": ["help", "you", "some"],
            "I need help": ["please", "now", "thanks"],
            "I need you": ["please", "now", "here"],
            "I need some": ["help"],
            "I want": ["help", "you"],
            "I want help": ["please", "now", "thanks"],
            "I want you": ["please", "now", "here"],
            "I am": ["okay", "fine", "good", "alright"],
            "I don't": ["need", "want", "think"],
            "I don't need": ["help", "anything", "it"],
            "I don't want": ["help", "anything", "it"],
            "I don't think": ["so"],
            "I can't": ["do", "reach", "move"],
            "I can't do": ["it"],
            "I can't reach": ["it"],
            "I can't move": ["now", "please", "thanks"],
            "Not": ["now", "yet", "today", "thanks"],
            "Not now": ["thanks", "please", "later"],
            "Please": ["help", "come", "wait"],
            "Please help": ["me", "now", "please"],
            "Please come": ["here", "now", "please"],
            "Please wait": ["please", "now", "here"],
            "Maybe": ["later", "yes", "no", "please"],
            "Maybe yes": ["please", "now", "thanks"],
            "Maybe no": ["thanks", "now", "today"],
            "Thanks": ["I'm", "now", "today"],
            "Thanks I'm": ["okay", "fine", "good", "alright"],
        ],
        closers: ["please", "now", "thanks", "today"])

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
            "I want to go out": ["now", "please", "today"],
            "I want to go bed": ["now", "please"],
            "I want to stay": ["here", "home", "inside", "now"],
            "I want to leave": ["now", "please", "today"],
            "I want to come": ["home", "inside", "now"],
            "I want home": ["now", "please", "today"],
            "I want out": ["now", "please", "today"],
            "I want outside": ["now", "please", "today"],
            "I would": ["like", "love", "rather"],
            "I would like": ["to", "home", "outside"],
            "I would like to": ["go", "stay", "leave"],
            "I would like to go": ["home", "outside", "back", "there", "inside"],
            "I would like to stay": ["here", "home", "inside", "now"],
            "I would like to leave": ["now", "please", "today"],
            "I would love": ["to", "home", "outside"],
            "I would love to": ["go", "stay", "leave"],
            "I would love to go": ["home", "outside", "back", "there", "inside"],
            "I would love to stay": ["here", "home", "inside", "now"],
            "I would love to leave": ["now", "please", "today"],
            "I would rather": ["go", "stay", "leave"],
            "I would rather go": ["home", "outside", "back", "there", "inside"],
            "I would rather stay": ["here", "home", "inside", "now"],
            "I would rather leave": ["now", "please", "today"],
            "I need": ["to", "home"],
            "I need home": ["now", "please", "today"],
            "I need to": ["go", "stay", "leave"],
            "I need to go": ["home", "outside", "back", "there", "inside"],
            "I need to stay": ["here", "home", "inside", "now"],
            "I need to leave": ["now", "please", "today"],
            "I like": ["home", "outside", "here"],
            "I don't": ["want", "know", "care"],
            "I don't want": ["to"],
            "I don't want to": ["go", "stay", "leave"],
            "I don't want to go": ["anywhere", "outside", "back", "there"],
            "I don't want to stay": ["here", "inside", "now"],
            "I don't want to leave": ["now", "please", "today"],
            "I don't know": ["now", "thanks", "today"],
            "I don't care": ["now", "thanks", "today"],
            "I'd": ["like", "love", "rather"],
            "I'd like": ["to", "home", "outside"],
            "I'd like to": ["go", "stay", "leave"],
            "I'd like to go": ["home", "outside", "back", "there", "inside"],
            "I'd like to stay": ["here", "home", "inside", "now"],
            "I'd like to leave": ["now", "please", "today"],
            "I'd love": ["to", "home", "outside"],
            "I'd love to": ["go", "stay", "leave"],
            "I'd love to go": ["home", "outside", "back", "there", "inside"],
            "I'd love to stay": ["here", "home", "inside", "now"],
            "I'd love to leave": ["now", "please", "today"],
            "I'd rather": ["go", "stay", "leave"],
            "I'd rather go": ["home", "outside", "back", "there", "inside"],
            "I'd rather stay": ["here", "home", "inside", "now"],
            "I'd rather leave": ["now", "please", "today"],
            "Home": ["please", "now", "today", "thanks"],
            "Outside": ["please", "now", "today"],
            "Bed": ["please", "now"],
            "Nowhere": ["thanks", "now", "today"],
            "Please": ["home", "now", "take"],
            "Please take": ["me"],
            "Please take me": ["home", "outside", "there"],
        ],
        closers: ["now", "please", "today", "thanks"])
}
