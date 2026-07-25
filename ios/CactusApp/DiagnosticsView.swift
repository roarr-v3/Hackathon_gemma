import Darwin
import SwiftUI

#if DEBUG
struct DiagnosticsSnapshot {
    var physicalFootprint: UInt64 = 0
    var peakPhysicalFootprint: UInt64 = 0
    var availableMemory: UInt64 = 0
    var virtualSize: UInt64 = 0
    var internalMemory: UInt64 = 0
    var compressedMemory: UInt64 = 0
    var graphicsMemory: UInt64 = 0
    var neuralMemory: UInt64 = 0
    var freeStorage: Int64 = 0
    var thermalState = "Unknown"
    var lowPowerMode = false

    var estimatedMemoryLimit: UInt64 {
        physicalFootprint + availableMemory
    }

    var memoryFraction: Double {
        guard estimatedMemoryLimit > 0 else { return 0 }
        return min(Double(physicalFootprint) / Double(estimatedMemoryLimit), 1)
    }

    static func capture() -> Self {
        var snapshot = Self()
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size
                / MemoryLayout<natural_t>.size
        )

        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(
                to: integer_t.self,
                capacity: Int(count)
            ) { reboundPointer in
                task_info(
                    mach_task_self_,
                    task_flavor_t(TASK_VM_INFO),
                    reboundPointer,
                    &count
                )
            }
        }

        if result == KERN_SUCCESS {
            snapshot.physicalFootprint = UInt64(info.phys_footprint)
            snapshot.peakPhysicalFootprint = UInt64(
                max(info.ledger_phys_footprint_peak, 0)
            )
            snapshot.virtualSize = UInt64(info.virtual_size)
            snapshot.internalMemory = UInt64(info.internal)
            snapshot.compressedMemory = UInt64(info.compressed)
            snapshot.graphicsMemory = UInt64(
                max(info.ledger_tag_graphics_footprint, 0)
            )
            snapshot.neuralMemory = UInt64(
                max(info.ledger_tag_neural_footprint, 0)
            )
        }

        snapshot.availableMemory = UInt64(os_proc_available_memory())
        let processInfo = ProcessInfo.processInfo
        snapshot.thermalState = processInfo.thermalState.displayName
        snapshot.lowPowerMode = processInfo.isLowPowerModeEnabled
        snapshot.freeStorage = (
            try? ModelStorage.modelsDirectory.resourceValues(
                forKeys: [.volumeAvailableCapacityForImportantUsageKey]
            ).volumeAvailableCapacityForImportantUsage
        ) ?? 0
        return snapshot
    }
}

struct DiagnosticsView: View {
    @ObservedObject var modelManager: ModelManager
    @State private var snapshot = DiagnosticsSnapshot.capture()

    var body: some View {
        NavigationStack {
            List {
                Section("Memory") {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Text("Memory pressure")
                            Spacer()
                            Text("\(Int(snapshot.memoryFraction * 100))%")
                                .foregroundStyle(memoryColor)
                        }
                        ProgressView(value: snapshot.memoryFraction)
                            .tint(memoryColor)
                    }
                    .padding(.vertical, 4)

                    metric("Physical footprint", snapshot.physicalFootprint)
                    metric("Peak footprint", snapshot.peakPhysicalFootprint)
                    metric("Available headroom", snapshot.availableMemory)
                    metric("Estimated app limit", snapshot.estimatedMemoryLimit)
                    metric("Virtual address space", snapshot.virtualSize)
                    metric("Internal", snapshot.internalMemory)
                    metric("Compressed", snapshot.compressedMemory)
                    metric("Metal / graphics", snapshot.graphicsMemory)
                    metric("Neural", snapshot.neuralMemory)
                }

                Section("Device") {
                    LabeledContent("Thermal state", value: snapshot.thermalState)
                    LabeledContent(
                        "Low Power Mode",
                        value: snapshot.lowPowerMode ? "On" : "Off"
                    )
                    LabeledContent(
                        "Free storage",
                        value: format(snapshot.freeStorage)
                    )
                }

                Section("Model") {
                    LabeledContent("Name", value: "Gemma 4 E2B CQ4")
                    LabeledContent(
                        "Status",
                        value: ModelStorage.isInstalled ? "Installed" : "Not installed"
                    )
                    if case let .installed(size) = modelManager.state {
                        LabeledContent("Disk usage", value: format(size))
                    }
                    LabeledContent(
                        "High-memory entitlement",
                        value: "Enabled"
                    )
                    LabeledContent(
                        "Extended virtual addressing",
                        value: "Disabled"
                    )
                }

                Section {
                    Text("Values update once per second. “Available headroom” is the dirty-memory allowance reported by iOS and can change while the app runs.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Diagnostics")
            .task {
                while !Task.isCancelled {
                    snapshot = DiagnosticsSnapshot.capture()
                    try? await Task.sleep(for: .seconds(1))
                }
            }
        }
    }

    private var memoryColor: Color {
        switch snapshot.memoryFraction {
        case 0.85...:
            .red
        case 0.7...:
            .orange
        default:
            .green
        }
    }

    private func metric(_ title: String, _ bytes: UInt64) -> some View {
        LabeledContent(title, value: format(Int64(clamping: bytes)))
    }

    private func format(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .memory)
    }
}

private extension ProcessInfo.ThermalState {
    var displayName: String {
        switch self {
        case .nominal:
            "Nominal"
        case .fair:
            "Fair"
        case .serious:
            "Serious"
        case .critical:
            "Critical"
        @unknown default:
            "Unknown"
        }
    }
}
#endif
