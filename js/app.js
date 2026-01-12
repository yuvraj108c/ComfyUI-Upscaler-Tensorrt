import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "UpscalerTensorrt.CustomResize",

    async nodeCreated(node) {
        if (node.comfyClass !== "UpscalerTensorrt") return;

        function setup() {
            const resizeToWidget = node.widgets.find(w => w.name === "resize_to");
            const resizeWidthWidget = node.widgets.find(w => w.name === "resize_width");
            const resizeHeightWidget = node.widgets.find(w => w.name === "resize_height");

            if (!resizeToWidget || !resizeWidthWidget || !resizeHeightWidget) {
                    // Widgets not ready yet → try again next frame
                    requestAnimationFrame(setup);
                    return;
            }

            function update(value) {
                const show = value === "custom";
                resizeWidthWidget.hidden = !show;
                resizeHeightWidget.hidden = !show;
                node.setSize([node.size[0], show ? 126 : 76]);
                node.setDirtyCanvas(true, true);
            }
            update(resizeToWidget.value);

            const orig = resizeToWidget.callback;
            resizeToWidget.callback = (v) => {
                update(v);
                orig?.(v);
            };
        }
        requestAnimationFrame(setup)
    }
});
