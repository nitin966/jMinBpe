import tokenpair.TokenPair;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * A Byte Pair Encoding tokenizer that handles regex splitting patterns and special tokens.
 * This tokenizer follows the algorithmic approach of the GPT tokenizer.
 */
public class RegexTokenizer extends Tokenizer {

    private static final String GPT2_SPLIT_PATTERN = "'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
    private static final String GPT4_SPLIT_PATTERN = "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+";

    private final String pattern;
    private final Pattern compiledPattern;
    private Map<String, Integer> specialTokens;
    private Map<Integer, String> inverseSpecialTokens;
    protected Map<TokenPair, Integer> merges;
    protected Map<Integer, byte[]> vocab;

    /**
     * Constructs a new RegexTokenizer with an optional pattern.
     *
     * @param pattern The regex pattern to use for splitting text. If null, GPT4_SPLIT_PATTERN is used.
     */
    public RegexTokenizer(String pattern) {
        super();
        this.pattern = (pattern == null) ? GPT4_SPLIT_PATTERN : pattern;
        this.compiledPattern = Pattern.compile(this.pattern);
        this.specialTokens = new HashMap<>();
        this.inverseSpecialTokens = new HashMap<>();
        this.merges = new HashMap<>();
        this.vocab = new HashMap<>();
    }

    /**
     * Trains the tokenizer on the given text to create a vocabulary of the specified size.
     *
     * @param text The training text.
     * @param vocabSize The desired vocabulary size (must be at least 256).
     * @param verbose Whether to print progress information.
     */
    @Override
    public void train(String text, int vocabSize, boolean verbose) {
        if (vocabSize < 256) {
            throw new IllegalArgumentException("Vocab size must be at least 256");
        }
        int numMerges = vocabSize - 256;

        List<String> textChunks = findAll(compiledPattern, text);
        List<List<Integer>> ids = textChunks.stream()
            .map(ch -> new ArrayList<>(toByteList(ch.getBytes(StandardCharsets.UTF_8))))
            .collect(Collectors.toList());

        Map<TokenPair, Integer> merges = new HashMap<>();
        Map<Integer, byte[]> vocab = initializeVocab();

        for (int i = 0; i < numMerges; i++) {
            Map<TokenPair, Integer> stats = new HashMap<>();
            for (List<Integer> chunkIds : ids) {
                getStats(chunkIds, stats);
            }

            TokenPair bestPair = stats.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElseThrow(() -> new IllegalStateException("No pairs found"));

            int idx = 256 + i;
            ids = ids.stream()
                .map(chunkIds -> merge(chunkIds, bestPair, idx))
                .collect(Collectors.toList());

            merges.put(bestPair, idx);
            vocab.put(idx, concatenateBytes(vocab.get(bestPair.getFirst()), vocab.get(bestPair.getSecond())));

            if (verbose) {
                System.out.printf("merge %d/%d: %s -> %d (%s) had %d occurrences%n",
                    i + 1, numMerges, bestPair, idx, new String(vocab.get(idx), StandardCharsets.UTF_8), stats.get(bestPair));
            }
        }

        this.merges = merges;
        this.vocab = vocab;
    }

    /**
     * Registers special tokens with the tokenizer.
     *
     * @param specialTokens A map of special tokens to their corresponding IDs.
     */
    public void registerSpecialTokens(Map<String, Integer> specialTokens) {
        this.specialTokens = new HashMap<>(specialTokens);
        this.inverseSpecialTokens = specialTokens.entrySet().stream()
            .collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey));
    }

    /**
     * Decodes a list of token IDs back into text.
     *
     * @param ids The list of token IDs to decode.
     * @return The decoded text.
     */
    @Override
    public String decode(List<Integer> ids) {
        List<byte[]> partBytes = new ArrayList<>();
        for (int idx : ids) {
            if (vocab.containsKey(idx)) {
                partBytes.add(vocab.get(idx));
            } else if (inverseSpecialTokens.containsKey(idx)) {
                partBytes.add(inverseSpecialTokens.get(idx).getBytes(StandardCharsets.UTF_8));
            } else {
                throw new IllegalArgumentException("Invalid token id: " + idx);
            }
        }
        byte[] textBytes = concatenateByteArrays(partBytes);
        return new String(textBytes, StandardCharsets.UTF_8);
    }

    /**
     * Encodes a chunk of text into token IDs.
     *
     * @param textBytes The text to encode as a byte array.
     * @return A list of token IDs.
     */
    protected List<Integer> encodeChunk(byte[] textBytes) {
        List<Integer> ids = new ArrayList<>(toByteList(textBytes));
        while (ids.size() >= 2) {
            Map<TokenPair, Integer> stats = getStats(ids);
            TokenPair bestPair = stats.entrySet().stream()
                .min(Comparator.comparingInt(e -> merges.getOrDefault(e.getKey(), Integer.MAX_VALUE)))
                .map(Map.Entry::getKey)
                .orElse(null);

            if (bestPair == null || !merges.containsKey(bestPair)) {
                break;
            }

            int idx = merges.get(bestPair);
            ids = merge(ids, bestPair, idx);
        }
        return ids;
    }

    /**
     * Encodes text ignoring any special tokens.
     *
     * @param text The text to encode.
     * @return A list of token IDs.
     */
    public List<Integer> encodeOrdinary(String text) {
        List<String> textChunks = findAll(compiledPattern, text);
        List<Integer> ids = new ArrayList<>();
        for (String chunk : textChunks) {
            byte[] chunkBytes = chunk.getBytes(StandardCharsets.UTF_8);
            ids.addAll(encodeChunk(chunkBytes));
        }
        return ids;
    }

    /**
     * Encodes text, handling special tokens based on the specified allowed special tokens.
     *
     * @param text The text to encode.
     * @param allowedSpecial Specifies how to handle special tokens.
     * @return A list of token IDs.
     */
    public List<Integer> encode(String text, String allowedSpecial) {
        Map<String, Integer> special = determineSpecialTokens(allowedSpecial, text);
        if (special.isEmpty()) {
            return encodeOrdinary(text);
        }

        String specialPattern = "(" + String.join("|", special.keySet().stream().map(Pattern::quote).toArray(String[]::new)) + ")";
        String[] specialChunks = text.split(specialPattern, -1);
        List<Integer> ids = new ArrayList<>();

        for (int i = 0; i < specialChunks.length; i++) {
            if (i % 2 == 0) {
                ids.addAll(encodeOrdinary(specialChunks[i]));
            } else if (special.containsKey(specialChunks[i])) {
                ids.add(special.get(specialChunks[i]));
            }
        }

        return ids;
    }

    private Map<String, Integer> determineSpecialTokens(String allowedSpecial, String text) {
        switch (allowedSpecial) {
            case "all":
                return new HashMap<>(specialTokens);
            case "none":
                return new HashMap<>();
            case "none_raise":
                for (String token : specialTokens.keySet()) {
                    if (text.contains(token)) {
                        throw new IllegalArgumentException("Special token found in text: " + token);
                    }
                }
                return new HashMap<>();
            default:
                if (allowedSpecial.startsWith("set:")) {
                    Set<String> allowed = new HashSet<>(Arrays.asList(allowedSpecial.substring(4).split(",")));
                    return specialTokens.entrySet().stream()
                        .filter(e -> allowed.contains(e.getKey()))
                        .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
                }
                throw new IllegalArgumentException("Unrecognized allowed_special: " + allowedSpecial);
        }
    }

    private List<String> findAll(Pattern pattern, String text) {
        List<String> matches = new ArrayList<>();
        Matcher matcher = pattern.matcher(text);
        while (matcher.find()) {
            matches.add(matcher.group());
        }
        return matches;
    }

    private List<Integer> toByteList(byte[] bytes) {
        List<Integer> list = new ArrayList<>(bytes.length);
        for (byte b : bytes) {
            list.add((int) b & 0xFF);
        }
        return list;
    }

    private byte[] concatenateByteArrays(List<byte[]> arrays) {
        int totalLength = arrays.stream().mapToInt(arr -> arr.length).sum();
        byte[] result = new byte[totalLength];
        int offset = 0;
        for (byte[] array : arrays) {
            System.arraycopy(array, 0, result, offset, array.length);
            offset += array.length;
        }
        return result;
    }

    private Map<Integer, byte[]> initializeVocab() {
        Map<Integer, byte[]> vocab = new HashMap<>();
        for (int i = 0; i < 256; i++) {
            vocab.put(i, new byte[]{(byte) i});
        }
        return vocab;
    }

    private byte[] concatenateBytes(byte[] a, byte[] b) {
        byte[] result = new byte[a.length + b.length];
        System.arraycopy(a, 0, result, 0, a.length);
        System.arraycopy(b, 0, result, a.length, b.length);
        return result;
    }
}
