/**
 * @fileoverview This file contains the core service functions for interacting with the Google Gemini API.
 * It handles both the generation of comic scene prompts from a story (text generation)
 * and the generation of images for each of those prompts (image generation).
 * This version enhances safety filter error reporting for image generation and retries for 500 errors.
 * It also includes significantly improved prompting for character/style consistency and "4K" visual quality,
 * with hyper-strict character sheet enforcement, contextual adherence, and negative prompting against distortions.
 * Adds support for selecting text model, caption placement, and specific instructions for transgender character portrayal.
 * Further refines instructions for UNYIELDING facial consistency for characters across panels.
 * Adds fixed seed for image generation.
 */

import {
  GoogleGenAI,
  GenerateContentResponse as SDKGenerateContentResponse,
  GenerateImagesResponse as SDKGenerateImagesResponse,
  Modality,
  HarmCategory,
  HarmProbability,
} from "@google/genai";
import {
  ComicPanelData,
  StoryInputOptions,
  AspectRatio,
  ImageGenerationModel,
  TextGenerationModel,
  CaptionPlacement,
  ComicStyle,
  ComicEra,
  CharacterSheetDetails // New type for better structure
} from '../types';
import { FIXED_IMAGE_SEED } from '../constants'; // Import fixed seed

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

interface SafetyRating {
  category: HarmCategory;
  probability: HarmProbability;
  blocked?: boolean;
}

interface ScenePromptOutput {
  scene_number: number;
  scene_description_for_prompt: string;
  image_prompt: string;
  caption: string | null;
  dialogues: string[];
}

interface LLMSceneResponse {
  characterCanon?: Record<string, CharacterSheetDetails>; // For comprehensive character definitions
  scenes: ScenePromptOutput[];
}


/**
 * Generates a structured list of comic panel data from a user-provided story.
 * This function includes safety policy error handling and detailed instructions for consistency and quality.
 * @param apiKey - The user's Google Gemini API key.
 * @param options - The story input and comic customization options.
 * @returns A promise that resolves to an array of ComicPanelData objects.
 */
export const generateScenePrompts = async (apiKey: string, options: StoryInputOptions): Promise<ComicPanelData[]> => {
  if (!apiKey) throw new Error("API Key is required to generate scene prompts.");
  const ai = new GoogleGenAI({ apiKey });
  const { story, style, era, includeCaptions, numPages, aspectRatio, textModel, captionPlacement } = options;

  let aspectRatioDescription = "1:1 square";
  if (aspectRatio === AspectRatio.LANDSCAPE) {
    aspectRatioDescription = "16:9 landscape";
  } else if (aspectRatio === AspectRatio.PORTRAIT) {
    aspectRatioDescription = "9:16 portrait";
  }

  let captionDialogueInstruction = '';
  if (includeCaptions) {
    if (captionPlacement === CaptionPlacement.IN_IMAGE) {
      captionDialogueInstruction = `
IMPORTANT FOR CAPTIONS/DIALOGUES: If this scene has captions or dialogues, they MUST be incorporated directly into the 'image_prompt' itself, described as text elements visually present within the comic panel (e.g., 'a speech bubble above Zara’s head contains the text \"Let's go!\"', 'a rectangular yellow caption box at the bottom of the panel reads: \"Meanwhile, across town...\"').
The 'caption' and 'dialogues' fields in the JSON output for this panel MUST then be null or an empty array respectively. This is CRITICAL for embedding text in images.`;
    } else { // IN_UI
      captionDialogueInstruction = `
The "caption" field should contain a concise narrative caption for the scene. If no caption is appropriate, it should be null or an empty string.
The "dialogues" field should be an array of strings, where each string is a line of dialogue formatted as "CharacterName: \"Dialogue line\"". If no dialogue, it's an empty array.`;
    }
  } else {
    captionDialogueInstruction = `
Since captions and dialogues are disabled for this comic, the "caption" field in the JSON output MUST be null, and the "dialogues" field MUST be an empty array for all scenes.`;
  }


  const systemInstruction = `You are an AI assistant specialized in creating highly consistent and contextually accurate comic book scripts.
Your task is to break down the provided story into exactly ${numPages} scenes.
The output MUST be a single, valid JSON object containing two keys: "characterCanon" and "scenes".
"characterCanon" MUST be an object mapping character names to their detailed character sheets (see definition below).
"scenes" MUST be an array of objects, each representing a scene with these keys: "scene_number", "scene_description_for_prompt", "image_prompt", "caption", and "dialogues".
${captionDialogueInstruction}

**DEFINITION OF "characterCanon" OBJECT:**
A JSON object where keys are character names (e.g., "Zara", "The Guard"). Each value MUST be an object with the following keys, providing an exhaustive and UNCHANGING reference for visual consistency:
  *   "IVAP": "Internal Visual Anchor Phrase". A unique, highly concise summary of their MOST IMMUTABLE core visual features. For YOUR internal reference and prompt generation. (e.g., "ZARA_BLUE_SPIKY_HAIR_GREEN_EYES_SCAR").
  *   "appearance": A highly detailed description of their physical appearance: hairstyle, hair color, eye color and shape, facial features (e.g., nose shape, jawline, cheekbones, chin), specific facial marks (scars, moles), approximate age, build, height. This description is CRITICAL and MUST be the absolute visual canon.
  *   "attire": A highly detailed description of their typical clothing, including colors, styles, materials, and any accessories. This is also canonical unless explicitly changed by the story for a specific scene.
  *   "genderIdentityNote": (Optional) If the character is transgender, describe any specific portrayals to ensure respectful and accurate representation as per their identity, focusing on feminine features as per the story's intent unless a disguise is specified.

**EXAMPLE Character Sheet Entry:**
{
  "ZARA": {
    "IVAP": "ZARA_BLUE_SPIKY_HAIR_GREEN_EYES_SCAR",
    "appearance": "Young woman, early 20s, athletic build, approx 5'7\". Hair: spiky short blue hair, meticulously styled with a slight side part. Eyes: bright, piercing green, almond-shaped. Face: sharp nose, high cheekbones, strong defined jawline, and a small, very faint, almost invisible scar running vertically just above her left eyebrow. Determined and intense default expression.",
    "attire": "Well-worn dark brown leather jacket with prominent silver zippers and a slightly frayed collar, always worn over a plain, dark grey, fitted t-shirt. Ripped black skinny jeans with one large rip across the right knee. Black, heavy-duty combat boots, often scuffed, with distinctive red laces.",
    "genderIdentityNote": "Portrayed as a woman with feminine facial features and build."
  },
  "GUARD": {
    "IVAP": "GUARD_HEAVY_ARMOR_ANONYMOUS_FACE",
    "appearance": "Man, late 30s, burly build, approx 6'0\". Face is obscured by a full-face metal helmet with a simple visor slot. No visible hair or features.",
    "attire": "Full suit of dark, weathered plate armor with intricate etchings. A thick leather belt with pouches. Heavy steel gauntlets and boots. No visible skin except possibly through helmet visor.",
    "genderIdentityNote": null
  }
}

CRITICAL INSTRUCTIONS FOR CHARACTER CONSISTENCY, STORY CONTEXT, AND VISUAL QUALITY:

**PHASE 1: DEEP STORY ANALYSIS & CHARACTER DEFINITION (MANDATORY FIRST STEP)**
1.  **Thorough Story Comprehension:** After reviewing the ENTIRE story, identify ALL recurring characters and key visual elements.
2.  **Character Sheet Creation (CRITICAL):** For EACH character, create a comprehensive entry in the "characterCanon" object. This is your ABSOLUTE CANONICAL SOURCE OF TRUTH for ALL visual details.
    *   **UNYIELDING FACIAL CONSISTENCY:** The "appearance" field MUST detail facial structure, eye shape/color, nose shape, mouth, and unique marks (scars, moles) with the highest possible precision. These details are **CRITICALLY IMPORTANT** and MUST be maintained consistently across panels.
    *   **STYLE AND ERA INTEGRATION:** The chosen style "${style}" and era "${era}" must deeply influence the descriptive language and artistic direction. Incorporate visual cues specific to these choices into the character descriptions themselves (e.g., if "Vintage" era and "Cartoon" style, describe attire and features accordingly).
    *   **TRANSGENDER PORTRAYAL:** If a character is transgender, their "genderIdentityNote" and accompanying "appearance" and "attire" descriptions MUST reflect respectful and accurate portrayals consistent with their identity and the story's intent, prioritizing feminine features unless a disguise is explicitly described for a scene.

**PHASE 2: PANEL SCRIPT GENERATION (APPLY WITH UNYIELDING PRECISION)**

3.  **Unyielding Character Consistency (CRITICAL & NON-NEGOTIABLE):** For EACH "image_prompt" you generate:
    *   **START WITH THE CANON:** Begin the prompt with the character's "IVAP" and their full "appearance" and "attire" details from the "characterCanon" object.
    *   **FORMAT EXAMPLE:** "Character: ZARA_BLUE_SPIKY_HAIR_GREEN_EYES_SCAR (Zara), young woman, early 20s, athletic build, approx 5'7\", spiky short blue hair, meticulously styled with a slight side part, bright piercing green almond-shaped eyes, sharp nose, high cheekbones, strong defined jawline, small faint scar above left eyebrow. Wears dark brown leather jacket over black t-shirt, ripped black skinny jeans, combat boots with red laces. Scene: Zara confronts a heavily armored guard at the city gate. Mood: tense and dramatic. Aspect Ratio: ${aspectRatioDescription}."
    *   **MAINTAIN FACIAL IDENTITY LOCK:** The character's **facial identity** (overall facial structure, specific features like eye shape/color, nose shape, mouth, jawline, chin, forehead, and any unique marks like scars, moles, or freckles) MUST be **PERFECTLY AND UNERRINGLY REPLICATED** as described in their "characterCanon" entry. The face must be the **EXACT SAME FACE** in every image of this character. The ONLY exception is if the story *explicitly* describes a disguise that alters these features or a permanent, story-driven transformation *for that specific panel*.
    *   **NO SUMMARIZING, PARAPHRASING, OR OMITTING:** Use the character's canon details verbatim.

4.  **Strict Contextual Adherence & Narrative Continuity:**
    *   Each "image_prompt" must be a DIRECT VISUAL TRANSLATION of the events, character emotions, interactions, and environment AS DESCRIBED FOR THAT SPECIFIC SCENE.
    *   Mentally review story context. Ensure character states (injuries, fatigue, dirt, items held, attire changes IF STORY-DRIVEN) are accurately reflected.

5.  **"Immaculate Quality" Descriptors:** ALL "image_prompt" fields MUST aim for exceptionally high visual quality. Include terms such as: 'ultra-detailed', 'hyper-realistic textures (even for stylized art)', 'cinematic lighting setup', 'studio-quality lighting', 'global illumination', 'volumetric lighting', 'tack-sharp focus on subjects', 'professional digital painting', 'masterpiece composition', 'intricate environment details', 'vivid and harmonious color palette', 'flawless rendering', 'no digital artifacts'.

6.  **Scene Composition:** Each "image_prompt" should clearly describe the scene setting, character actions/poses, emotions, camera angle (e.g., 'dramatic close-up'), and overall mood.

Story to process:
---
${story}
---
CRITICAL OUTPUT FORMATTING:
1.  Produce ONLY the raw JSON object as your response. Do NOT include any explanatory text, markdown code fences (like \`\`\`json), or any other characters before or after the JSON object.
2.  All string values within the generated JSON MUST be properly escaped to ensure JSON validity. This includes, but is not limited to:
    *   Double quotes (\") inside strings must be escaped as \\".
    *   Backslashes (\\) inside strings must be escaped as \\\\.
    *   Newlines inside strings must be escaped as \\n.
    *   Tabs inside strings must be escaped as \\t.
3.  Pay meticulous attention to escaping requirements when incorporating text from the story or character sheets into fields like "image_prompt", "caption", and dialogue "line"s.
4.  Ensure the entire output is a single, complete, and syntactically correct JSON object conforming to the structure specified earlier.
`;

  try {
    const result: SDKGenerateContentResponse = await ai.models.generateContent({
      model: textModel, // Use the selected text model
      contents: [{ role: 'USER', parts: [{ text: systemInstruction }] }],
      config: { responseMimeType: "application/json" }
    });

    if (result.promptFeedback?.blockReason) {
      throw new Error(`Your story was blocked by content policies using model ${textModel} (${result.promptFeedback.blockReason}). Please revise your story.`);
    }
    
    const responseText = result.text;
    if (!responseText) {
      console.error(`API call for scene prompts (model: ${textModel}) returned invalid structure or no text:`, JSON.stringify(result, null, 2));
      throw new Error(`API response (model: ${textModel}) was malformed or did not contain expected text content.`);
    }

    let jsonStr = responseText.trim();
    const fenceRegex = /^```(\w*)?\s*\n?(.*?)\n?\s*```$/s;
    const match = jsonStr.match(fenceRegex);
    if (match && match[2]) {
      jsonStr = match[2].trim();
    }
    
    const parsedData: LLMSceneResponse = JSON.parse(jsonStr);

    // Validate required keys
    if (!parsedData || typeof parsedData !== 'object' || !Array.isArray(parsedData.scenes)) {
        throw new Error(`API response (model: ${textModel}) did not contain the expected JSON structure. Missing 'scenes' array or 'characterCanon'.`);
    }
    if (!parsedData.characterCanon || typeof parsedData.characterCanon !== 'object') {
        throw new Error(`API response (model: ${textModel}) did not contain the expected 'characterCanon' object.`);
    }

    const characterCanon = parsedData.characterCanon;
    const scenes = parsedData.scenes;

    return scenes.map((panel, index) => {
      let finalCaption = null;
      let finalDialogues: string[] = [];

      if (includeCaptions) {
        if (captionPlacement === CaptionPlacement.IN_UI) {
          finalCaption = panel.caption || "";
          finalDialogues = Array.isArray(panel.dialogues)
            ? panel.dialogues.map((d: any) => {
                if (typeof d === 'string') return d; // Already formatted or simple string
                if (d && d.character && d.line) return `${d.character}: "${d.line}"`; // Object format
                return String(d); // Fallback
              }).filter(Boolean)
            : [];
        }
      }
      
      // Pass character canon to panel for potential image prompt construction
      // If captionPlacement is IN_IMAGE, the LLM should have already embedded this in panel.image_prompt
      // If IN_UI, we still pass it to image prompt for consistency.
      // The image generation function WILL need this data, so we pass it as scene_description_for_prompt
      // or a dedicated field if needed. For now, let's assume image_prompt from LLM is detailed.
      // If we were to construct image_prompt here based on canon, we'd need to pass canon.

      return {
        scene_number: panel.scene_number || index + 1,
        // The image_prompt from the LLM is now expected to be a structured JSON string
        image_prompt: panel.image_prompt || "No prompt generated for this scene.",
        caption: finalCaption,
        dialogues: finalDialogues,
        scene_description_for_prompt: JSON.stringify({ // Store character canon and scene details
            characters: characterCanon,
            scene: panel.scene_description_for_prompt,
            mood: "default", // Placeholder, could be inferred
            // Add other context like aspectRatioDescription if needed for image generation
        })
      };
    });

  } catch (error) {
    const message = error instanceof Error ? error.message : "An unknown error occurred.";
    if (message.toLowerCase().includes("api key not valid") || message.toLowerCase().includes("permission denied")) {
        throw new Error(`Failed to generate scene prompts (model: ${textModel}) due to an API key issue: ${message}. Check your API key and permissions.`);
    }
    if (error instanceof SyntaxError || message.toLowerCase().includes("json") || message.toLowerCase().includes("unexpected token") || message.toLowerCase().includes("malformed")) {
      let detailedMessage = `Failed to parse scene prompts from API response (model: ${textModel}): ${message}. `;
      detailedMessage += `This can happen if the story is too long or requests too many pages, leading to an incomplete/malformed response. `;
      detailedMessage += `Try reducing pages or simplifying story.`;
      console.error("Original JSON parsing error details:", error);
      throw new Error(detailedMessage);
    }
    throw new Error(`Failed to generate scene prompts (model: ${textModel}). Error: ${message}. Ensure API key is valid and model is accessible.`);
  }
};


/**
 * Generates a single image for a given prompt, using the correct API method and configuration.
 * Includes enhanced prompting for consistency and quality, and negative prompting against distortions.
 * Uses a fixed seed for potentially more consistent image output.
 * @param apiKey - The user's Google Gemini API key.
 * @param initialImagePrompt - The detailed text prompt for the image (potentially includes character canon from LLM).
 * @param inputAspectRatio - The desired aspect ratio for the image.
 * @param imageModelName - The identifier for the image generation model.
 * @param style - The comic style (e.g., ComicStyle.ANIME).
 * @param era - The comic era (e.g., ComicEra.FUTURISTIC).
 * @returns A promise that resolves to a base64-encoded image data URL.
 */
export const generateImageForPrompt = async (
  apiKey: string,
  initialImagePrompt: string, // This prompt already contains character canon from generateScenePrompts
  inputAspectRatio: AspectRatio,
  imageModelName: ImageGenerationModel | string,
  style: ComicStyle | string, 
  era: ComicEra | string     
): Promise<string> => {
  if (!apiKey) throw new Error("API Key is required for image generation.");
  const ai = new GoogleGenAI({ apiKey });

  let apiAspectRatioValue: "1:1" | "9:16" | "16:9";
  switch (inputAspectRatio) {
    case AspectRatio.SQUARE: apiAspectRatioValue = "1:1"; break;
    case AspectRatio.PORTRAIT: apiAspectRatioValue = "9:16"; break;
    case AspectRatio.LANDSCAPE: apiAspectRatioValue = "16:9"; break;
    default: apiAspectRatioValue = "1:1";
  }

  // --- Constructing the SUPER PROMPT for Image Generation ---
  // The initialImagePrompt from generateScenePrompts should already contain character canon and specific scene details.
  // We will now layer on the MANDATORY artistic directives and consistency checks.

  let sceneDetails: any = {};
  try {
      // Attempt to parse the detailed prompt which should include characterCanon and scene description
      // We assume the format created in generateScenePrompts: { characters: {...}, scene: "...", ...}
      const parsedPrompt = JSON.parse(initialImagePrompt);
      if (parsedPrompt && typeof parsedPrompt === 'object') {
          sceneDetails = parsedPrompt;
      } else {
          // If it's just a string prompt, try to extract info or just use it as is.
          // For simplicity, we'll assume the LLM already structured it well if it's parsable JSON.
          // If it's not JSON, it might be a simpler string prompt.
          console.warn("Image prompt was not structured JSON, falling back to string interpretation.");
      }
  } catch (e) {
      console.warn("Could not parse initial image prompt JSON, treating as plain string.", e);
      // Fallback: if it's not JSON, treat it as a plain string prompt
      sceneDetails = { scene: initialImagePrompt };
  }

  const characterSections: string[] = [];
  if (sceneDetails.characters) {
      for (const characterName in sceneDetails.characters) {
          const char = sceneDetails.characters[characterName];
          // Ensure robust handling if character details are missing
          const ivap = char.IVAP ? `${char.IVAP} (${characterName})` : characterName;
          const description = char.description || "detailed description unavailable";
          characterSections.push(`Character: ${ivap}, ${description}.`);
      }
  }

  const sceneContextParts = [];
  if (sceneDetails.scene) sceneContextParts.push(`Scene: ${sceneDetails.scene}.`);
  if (sceneDetails.mood) sceneContextParts.push(`Mood: ${sceneDetails.mood}.`);
  // Add any other relevant details from sceneDetails here
  const sceneContext = sceneContextParts.join(' ');


  const augmentedPrompt = `
**[CHARACTER CANON AND SCENE CONTEXT - MANDATORY START]**
${characterSections.join('\n')}
${sceneContext}
**[END OF CHARACTER CANON AND SCENE CONTEXT]**

**[MANDATORY ARTISTIC DIRECTIVES AND UNYIELDING CONSISTENCY RULES]**
- **Overall Style:** Apply the "${style}" style with absolute fidelity. Ensure all elements (color palette, line art, rendering, composition) strictly adhere to this style.
- **Era Integration:** The scene MUST meticulously reflect the "${era}" period. Include era-appropriate details in backgrounds, props, and character attire if not explicitly superseded by character-specific attire.
- **Visual Quality:** Render with extreme detail, aiming for cinematic quality. Use ultra-high resolution aesthetics (e.g., 4K/8K equivalent). Implement advanced lighting techniques like volumetric lighting, rim lighting, and global illumination where fitting for the mood. Ensure subjects are tack-sharp, with intricate textures and environments. Aim for masterpiece-level digital art. Flawless rendering is essential.
- **[CRITICAL - FACIAL IDENTITY LOCK]:** The described facial structure, specific features (eyes, nose, mouth, jawline, unique marks like scars or moles), and hairstyle of ALL characters MUST remain **IDENTICALLY THE SAME** across all generated panels, as if rendered by the same artist with perfect recall. The character's face is the **ABSOLUTE VISUAL ANCHOR**. Only alter features if the scene explicitly dictates a disguise or transformation that modifies these specific elements.
- **[CRITICAL - CONSISTENCY]:** Character attire, accessories, and general appearance MUST also match the provided canon descriptions EXACTLY, unless the scene explicitly calls for a change.
- **[CRITICAL - TRANSGENDER REPRESENTATION]:** If a character is described as transgender, ensure their portrayal is consistently feminine in features and presentation, respecting their identity as per the character canon, unless an explicit disguise is part of this specific scene's prompt.
- **[CRITICAL - TEXT EMBEDDING]:** If the scene context implies text within the image (speech bubbles, signs, captions), render it CLEARLY and LEGIBLY as a visual element.

**[NEGATIVE PROMPTS - ABSOLUTELY AVOID]**
- **NEVER DEVIATE FACIAL FEATURES:** Do not alter character's established facial structure, eye shape/color, nose, mouth, jawline, scars, moles, or hair color/style from the provided descriptions.
- **AVOID VISUAL INCONSISTENCY:** Do not introduce stylistic shifts, rendering errors, pixelation, or digital artifacts.
- **NO WRONG ANATOMY:** Avoid malformed limbs, incorrect number of fingers, or distorted perspectives.
- **CLEAN TEXT:** Ensure any text within the image is perfectly readable and accurate.
- **BACKGROUND INTEGRITY:** Do not use generic or mismatched backgrounds; ensure they fit the scene and era.
- **EMOTIONAL ACCURACY:** Character expressions and poses must match the described mood and actions.
- **NO ANachronisms:** Strictly adhere to the specified era for all elements.
`;

  const maxRetries = 2;
  let retries = 0;
  const baseDelayMs = 2000;

  while (retries <= maxRetries) {
    try {
      // Gemini Flash Image Generation
      if (imageModelName === ImageGenerationModel.GEMINI_2_FLASH_IMG) {
        const result: SDKGenerateContentResponse = await ai.models.generateContent({
          model: imageModelName,
          contents: [{ role: 'USER', parts: [{ text: augmentedPrompt }] }], 
          config: {
            responseModalities: [Modality.TEXT, Modality.IMAGE],
            seed: FIXED_IMAGE_SEED, // Added fixed seed
          }
        });
        
        const candidate = result.candidates?.[0];
        if (candidate) {
          if (candidate.finishReason === "SAFETY" || candidate.finishReason === "IMAGE_SAFETY") {
            let safetyMessage = `Image generation was blocked by safety filters (Reason: ${candidate.finishReason}).`;
            const typedSafetyRatings = candidate.safetyRatings as SafetyRating[] | undefined;
            const blockedRating = typedSafetyRatings?.find(r => r.blocked === true);
            if (blockedRating) {
              safetyMessage += ` Details: Category ${blockedRating.category}, Probability ${blockedRating.probability}.`;
            }
            safetyMessage += " Please try a different prompt or adjust story content.";
            console.error("Image generation safety block details:", JSON.stringify(candidate, null, 2));
            throw new Error(safetyMessage);
          }

          const imagePart = candidate.content?.parts?.find(part => part.inlineData);
          if (imagePart?.inlineData?.data && imagePart.inlineData.mimeType) {
            return `data:${imagePart.inlineData.mimeType};base64,${imagePart.inlineData.data}`;
          }
        }

        console.error("Gemini Flash response did not contain image data or was unexpected:", JSON.stringify(result, null, 2));
        const textFeedbackParts = candidate?.content?.parts?.filter(part => part.text);
        const textFeedback = textFeedbackParts && textFeedbackParts.length > 0
          ? textFeedbackParts.map(p => p.text).join(' ')
          : "No explicit text feedback from API.";

        throw new Error(`API did not return an image for ${imageModelName}. Feedback: "${textFeedback}". This could be due to restrictive safety filters, an issue with the prompt, or an API problem. Full response logged to console.`);

      } 
      // Imagen models (or others that use generateImages)
      else { 
        const response: SDKGenerateImagesResponse = await ai.models.generateImages({
          model: imageModelName,
          prompt: augmentedPrompt,
          config: {
            numberOfImages: 1,
            outputMimeType: 'image/jpeg',
            aspectRatio: apiAspectRatioValue,
            seed: FIXED_IMAGE_SEED, // Added fixed seed
          },
        });

        // Check for API errors for Imagen
        if ((response as any).error || (response as any).code) {
             throw new Error(`Image generation failed for ${imageModelName}. API Error: ${JSON.stringify((response as any).error || (response as any).code)}`);
        }

        if (response.generatedImages?.[0]?.image?.imageBytes) {
          return `data:image/jpeg;base64,${response.generatedImages[0].image.imageBytes}`;
        } else {
          console.error(`No image data received from ${imageModelName} API. Response:`, JSON.stringify(response, null, 2));
          throw new Error(`No image data received from ${imageModelName} API. This could be due to safety filters or other issues. Full response logged to console.`);
        }
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      if (errorMessage.toLowerCase().includes("api key not valid") || errorMessage.toLowerCase().includes("permission denied")) {
        throw new Error(`Image generation failed for model '${imageModelName}' due to an API key issue: ${errorMessage}. Please check your API key and its permissions.`);
      }

      const isRateLimitError = errorMessage.toLowerCase().includes("429") ||
                               errorMessage.toUpperCase().includes("RESOURCE_EXHAUSTED") ||
                               errorMessage.toUpperCase().includes("RATE_LIMIT_EXCEEDED");

      const isInternalServerError = errorMessage.toLowerCase().includes("status: 500") ||
                                    errorMessage.toUpperCase().includes("INTERNAL ERROR") ||
                                    errorMessage.toUpperCase().includes("INTERNAL");


      if ((isRateLimitError || isInternalServerError) && retries < maxRetries) {
        retries++;
        const delayTime = baseDelayMs * Math.pow(2, retries - 1) + (Math.random() * 1000);
        console.warn(`API error for '${imageModelName}' (Type: ${isRateLimitError ? 'Rate Limit/Resource' : 'Internal Server'}). Retrying attempt ${retries}/${maxRetries} in ${Math.round(delayTime / 1000)}s... Error: ${errorMessage}`);
        await delay(delayTime);
      } else if (errorMessage.toLowerCase().includes("blocked by safety filters") || errorMessage.toLowerCase().includes("safety policy")) {
        throw error; // Propagate safety errors immediately
      } else {
        throw new Error(`API call failed for model '${imageModelName}' (attempt ${retries + 1}/${maxRetries + 1}). Original error: ${errorMessage}`);
      }
    }
  }
  throw new Error(`Failed to generate image after all retries for model '${imageModelName}'. Check console for details.`);
};
