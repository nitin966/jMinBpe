import tokenpair.TokenPair;
import utils.ByteUtils;
import utils.TokenUtils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A tokenizer class that implements Byte Pair Encoding (BPE) algorithm.
 * This class provides functionality to encode text into tokens and decode tokens back to text,
 * as well as train new vocabularies and save/load models.
 */
public class Tokenizer {
    protected Map<TokenPair, Integer> merges;
    private String pattern;
    private final Map<String, Integer> specialTokens;
    protected Map<Integer, byte[]> vocab;

    /**
     * Constructs a new Tokenizer with default settings.
     */
    public Tokenizer() {
        this.merges = new HashMap<>();
        this.pattern = "";
        this.specialTokens = new HashMap<>();
        this.vocab = buildVocab();
    }

    /**
     * Builds the vocabulary based on merges and special tokens.
     *
     * @return A map of token IDs to their byte representations.
     */
    private Map<Integer, byte[]> buildVocab() {
        Map<Integer, byte[]> newVocab = new HashMap<>(256);
        for (int i = 0; i < 256; i++) {
            newVocab.put(i, new byte[]{(byte) i});
        }
        merges.forEach((pair, idx) -> 
            newVocab.put(idx, ByteUtils.concatenate(newVocab.get(pair.getFirst()), newVocab.get(pair.getSecond())))
        );
        specialTokens.forEach((token, idx) -> 
            newVocab.put(idx, token.getBytes(StandardCharsets.UTF_8))
        );
        return newVocab;
    }

    /**
     * Trains the tokenizer on the given text to create a vocabulary of the specified size.
     *
     * @param text The training text.
     * @param vocabSize The desired vocabulary size.
     * @param verbose Whether to print progress information.
     */
    public void train(String text, int vocabSize, boolean verbose) {
        // Implementation of BPE training algorithm
        throw new UnsupportedOperationException("Train method not implemented");
    }

    /**
     * Encodes the given text into a list of token IDs.
     *
     * @param text The text to encode.
     * @return A list of token IDs.
     */
    public List<Integer> encode(String text) {
        // Implementation of encoding algorithm
        throw new UnsupportedOperationException("Encode method not implemented");
    }

    /**
     * Decodes the given list of token IDs back into text.
     *
     * @param ids The list of token IDs to decode.
     * @return The decoded text.
     */
    public String decode(List<Integer> ids) {
        // Implementation of decoding algorithm
        throw new UnsupportedOperationException("Decode method not implemented");
    }

    /**
     * Saves the current tokenizer model to files.
     *
     * @param filePrefix The prefix for the output files.
     * @throws IOException If an I/O error occurs.
     */
    public void save(String filePrefix) throws IOException {
        saveModel(filePrefix + ".model");
        saveVocab(filePrefix + ".vocab");
    }

    /**
     * Saves the model information to a file.
     *
     * @param fileName The name of the file to save the model to.
     * @throws IOException If an I/O error occurs.
     */
    private void saveModel(String fileName) throws IOException {
        try (PrintWriter writer = new PrintWriter(fileName, "UTF-8")) {
            writer.println("minbpe v1");
            writer.println(pattern);
            writer.println(specialTokens.size());
            specialTokens.forEach((token, idx) -> writer.println(token + " " + idx));
            merges.forEach((pair, idx) -> writer.println(pair.getFirst() + " " + pair.getSecond()));
        }
    }

    /**
     * Saves the vocabulary information to a file for human inspection.
     *
     * @param fileName The name of the file to save the vocabulary to.
     * @throws IOException If an I/O error occurs.
     */
    private void saveVocab(String fileName) throws IOException {
        try (PrintWriter writer = new PrintWriter(fileName, "UTF-8")) {
            Map<Integer, TokenPair> invertedMerges = new HashMap<>();
            merges.forEach((pair, idx) -> invertedMerges.put(idx, pair));

            vocab.forEach((idx, token) -> {
                String s = TokenUtils.renderToken(token);
                if (invertedMerges.containsKey(idx)) {
                    TokenPair pair = invertedMerges.get(idx);
                    String s0 = TokenUtils.renderToken(vocab.get(pair.getFirst()));
                    String s1 = TokenUtils.renderToken(vocab.get(pair.getSecond()));
                    writer.println("[" + s0 + "][" + s1 + "] -> [" + s + "] " + idx);
                } else {
                    writer.println("[" + s + "] " + idx);
                }
            });
        }
    }

    /**
     * Loads a tokenizer model from a file.
     *
     * @param modelFile The file to load the model from.
     * @throws IOException If an I/O error occurs.
     */
    public void load(String modelFile) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(modelFile))) {
            String version = reader.readLine().trim();
            if (!version.equals("minbpe v1")) {
                throw new IllegalStateException("Incorrect version");
            }

            pattern = reader.readLine().trim();
            int numSpecial = Integer.parseInt(reader.readLine().trim());
            
            loadSpecialTokens(reader, numSpecial);
            loadMerges(reader);
        }
        vocab = buildVocab();
    }

    /**
     * Loads special tokens from the reader.
     *
     * @param reader The BufferedReader to read from.
     * @param numSpecial The number of special tokens to read.
     * @throws IOException If an I/O error occurs.
     */
    private void loadSpecialTokens(BufferedReader reader, int numSpecial) throws IOException {
        for (int i = 0; i < numSpecial; i++) {
            String[] parts = reader.readLine().trim().split(" ");
            specialTokens.put(parts[0], Integer.parseInt(parts[1]));
        }
    }

    /**
     * Loads merges from the reader.
     *
     * @param reader The BufferedReader to read from.
     * @throws IOException If an I/O error occurs.
     */
    private void loadMerges(BufferedReader reader) throws IOException {
        String line;
        int idx = 256;
        while ((line = reader.readLine()) != null) {
            String[] parts = line.split(" ");
            int idx1 = Integer.parseInt(parts[0]);
            int idx2 = Integer.parseInt(parts[1]);
            merges.put(new TokenPair(idx1, idx2), idx++);
        }
    }

    protected void getStats(List<Integer> ids, Map<TokenPair, Integer> stats) {
        for (int i = 0; i < ids.size() - 1; i++) {
            TokenPair pair = new TokenPair(ids.get(i), ids.get(i + 1));
            stats.put(pair, stats.getOrDefault(pair, 0) + 1);
        }
    }

    protected Map<TokenPair, Integer> getStats(List<Integer> ids) {
        Map<TokenPair, Integer> stats = new HashMap<>();
        getStats(ids, stats);
        return stats;
    }

    protected List<Integer> merge(List<Integer> ids, TokenPair pair, int newId) {
        List<Integer> newIds = new ArrayList<>();
        for (int i = 0; i < ids.size(); i++) {
            if (i < ids.size() - 1 && ids.get(i).equals(pair.getFirst()) && ids.get(i + 1).equals(pair.getSecond())) {
                newIds.add(newId);
                i++;
            } else {
                newIds.add(ids.get(i));
            }
        }
        return newIds;
    }
}
