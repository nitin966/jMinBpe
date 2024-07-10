import tokenpair.TokenPair;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A minimal byte-level Byte Pair Encoding (BPE) tokenizer.
 * This implementation follows the algorithmic approach of the GPT tokenizer
 * but does not handle regular expression splitting patterns or special tokens.
 */
public class BasicTokenizer extends Tokenizer {

    /**
     * Constructs a new BasicTokenizer.
     */
    public BasicTokenizer() {
        super();
    }

    /**
     * Trains the tokenizer on the given text to create a vocabulary of the specified size.
     *
     * @param text The training text.
     * @param vocabSize The desired vocabulary size (must be at least 256).
     * @param verbose Whether to print progress information.
     * @throws IllegalArgumentException if vocabSize is less than 256.
     */
    @Override
    public void train(String text, int vocabSize, boolean verbose) {
        if (vocabSize < 256) {
            throw new IllegalArgumentException("Vocab size must be at least 256");
        }

        int numMerges = vocabSize - 256;
        byte[] textBytes = text.getBytes(StandardCharsets.UTF_8);
        List<Integer> ids = new ArrayList<>(textBytes.length);
        for (byte b : textBytes) {
            ids.add((int) b & 0xFF);
        }

        Map<TokenPair, Integer> merges = new HashMap<>();
        Map<Integer, byte[]> vocab = initializeVocab();

        for (int i = 0; i < numMerges; i++) {
            Map<TokenPair, Integer> stats = getStats(ids);
            TokenPair bestPair = findMostFrequentPair(stats);
            int newTokenId = 256 + i;

            ids = merge(ids, bestPair, newTokenId);
            merges.put(bestPair, newTokenId);
            vocab.put(newTokenId, concatenateBytes(vocab.get(bestPair.getFirst()), vocab.get(bestPair.getSecond())));

            if (verbose) {
                System.out.printf("merge %d/%d: %s -> %d (%s) had %d occurrences%n",
                        i + 1, numMerges, bestPair, newTokenId, 
                        new String(vocab.get(newTokenId), StandardCharsets.UTF_8), stats.get(bestPair));
            }
        }

        this.merges = merges;
        this.vocab = vocab;
    }

    /**
     * Decodes a list of token IDs back into text.
     *
     * @param ids The list of token IDs to decode.
     * @return The decoded text.
     */
    @Override
    public String decode(List<Integer> ids) {
        byte[] textBytes = ids.stream()
                              .map(this.vocab::get)
                              .reduce(new byte[0], this::concatenateBytes);
        return new String(textBytes, StandardCharsets.UTF_8);
    }

    /**
     * Encodes the given text into a list of token IDs.
     *
     * @param text The text to encode.
     * @return A list of token IDs.
     */
    @Override
    public List<Integer> encode(String text) {
        byte[] textBytes = text.getBytes(StandardCharsets.UTF_8);
        List<Integer> ids = new ArrayList<>(textBytes.length);
        for (byte b : textBytes) {
            ids.add((int) b & 0xFF);
        }

        boolean merged;
        do {
            merged = false;
            Map<TokenPair, Integer> stats = getStats(ids);
            TokenPair bestPair = findBestPair(stats);
            if (bestPair != null) {
                int idx = this.merges.get(bestPair);
                ids = merge(ids, bestPair, idx);
                merged = true;
            }
        } while (merged && ids.size() >= 2);

        return ids;
    }

    /**
     * Initializes the vocabulary with byte values.
     *
     * @return A map of token IDs to their byte representations.
     */
    private Map<Integer, byte[]> initializeVocab() {
        Map<Integer, byte[]> vocab = new HashMap<>(256);
        for (int i = 0; i < 256; i++) {
            vocab.put(i, new byte[]{(byte) i});
        }
        return vocab;
    }

    /**
     * Finds the most frequent pair in the statistics.
     *
     * @param stats The statistics of token pair occurrences.
     * @return The most frequent TokenPair.
     */
    private TokenPair findMostFrequentPair(Map<TokenPair, Integer> stats) {
        return stats.entrySet().stream()
                    .max(Map.Entry.comparingByValue())
                    .map(Map.Entry::getKey)
                    .orElseThrow(() -> new IllegalStateException("No pairs found"));
    }

    /**
     * Finds the best pair for merging based on the merge index.
     *
     * @param stats The statistics of token pair occurrences.
     * @return The best TokenPair for merging, or null if no merge is possible.
     */
    private TokenPair findBestPair(Map<TokenPair, Integer> stats) {
        return stats.keySet().stream()
                    .min(Comparator.comparingInt(pair -> this.merges.getOrDefault(pair, Integer.MAX_VALUE)))
                    .filter(this.merges::containsKey)
                    .orElse(null);
    }

    /**
     * Concatenates two byte arrays.
     *
     * @param a The first byte array.
     * @param b The second byte array.
     * @return The concatenated byte array.
     */
    private byte[] concatenateBytes(byte[] a, byte[] b) {
        byte[] result = new byte[a.length + b.length];
        System.arraycopy(a, 0, result, 0, a.length);
        System.arraycopy(b, 0, result, a.length, b.length);
        return result;
    }
}
